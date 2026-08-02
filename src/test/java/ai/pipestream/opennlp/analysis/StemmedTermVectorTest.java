/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package ai.pipestream.opennlp.analysis;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import ai.pipestream.opennlp.analysis.config.ServiceConfig;
import ai.pipestream.opennlp.analysis.pipeline.PipelineEnvironment;
import ai.pipestream.opennlp.analysis.v1.AnalysisOptions;
import ai.pipestream.opennlp.analysis.v1.AnalysisServiceGrpc;
import ai.pipestream.opennlp.analysis.v1.AnalyzeRequest;
import ai.pipestream.opennlp.analysis.v1.AnalyzeResponse;
import ai.pipestream.opennlp.analysis.v1.TermVectorOptions;
import io.grpc.ManagedChannel;
import io.grpc.Server;
import io.grpc.Status;
import io.grpc.StatusRuntimeException;
import io.grpc.inprocess.InProcessChannelBuilder;
import io.grpc.inprocess.InProcessServerBuilder;

/**
 * Stem-sourced term vectors: the stem is the term identity, occurrences keep
 * their original-text token spans, and the normalizer steps are ignored.
 */
class StemmedTermVectorTest {

  private Server server;
  private ManagedChannel channel;
  private AnalysisServiceGrpc.AnalysisServiceBlockingStub stub;

  @BeforeEach
  void setUp() throws Exception {
    final String name = InProcessServerBuilder.generateName();
    server = InProcessServerBuilder.forName(name).directExecutor()
        .addService(new AnalysisServiceImpl(PipelineEnvironment.empty(),
            new ServiceConfig(0, 1024 * 1024, null, null, null, null, null, null)))
        .build().start();
    channel = InProcessChannelBuilder.forName(name).directExecutor().build();
    stub = AnalysisServiceGrpc.newBlockingStub(channel);
  }

  @AfterEach
  void tearDown() {
    channel.shutdownNow();
    server.shutdownNow();
  }

  private static TermVectorOptions.Builder stemmed(TermVectorOptions.Mode mode) {
    return TermVectorOptions.newBuilder()
        .setEnabled(true)
        .setSource(TermVectorOptions.Source.SOURCE_STEMS)
        .setMode(mode);
  }

  @Test
  void porterGroupsRunningRunsRunUnderOneStem() {    final AnalyzeResponse response = stub.analyze(AnalyzeRequest.newBuilder()
        .setText("running runs run")
        .setOptions(AnalysisOptions.newBuilder()
            .setStemmer(AnalysisOptions.Stemmer.STEMMER_PORTER)
            .setTermVectors(stemmed(TermVectorOptions.Mode.MODE_FULL)))
        .build());

    assertThat(response.getTermVectorsCount()).isEqualTo(1);
    final ai.pipestream.opennlp.analysis.v1.TermVector vector = response.getTermVectors(0);
    assertThat(vector.getTerm()).isEqualTo("run");
    assertThat(vector.getFrequency()).isEqualTo(3);
    assertThat(vector.getOccurrencesList())
        .extracting(s -> s.getStart() + "-" + s.getEnd())
        .containsExactly("0-7", "8-12", "13-16");
  }

  @Test
  void scoringOnlyOmitsOccurrencesInStemMode() {
    final AnalyzeResponse response = stub.analyze(AnalyzeRequest.newBuilder()
        .setText("running runs run")
        .setOptions(AnalysisOptions.newBuilder()
            .setStemmer(AnalysisOptions.Stemmer.STEMMER_PORTER)
            .setTermVectors(stemmed(TermVectorOptions.Mode.MODE_SCORING_ONLY)))
        .build());

    assertThat(response.getTermVectorsCount()).isEqualTo(1);
    assertThat(response.getTermVectors(0).getTerm()).isEqualTo("run");
    assertThat(response.getTermVectors(0).getFrequency()).isEqualTo(3);
    assertThat(response.getTermVectors(0).getOccurrencesCount()).isZero();
  }

  @Test
  void stepsAreIgnoredForIdentityInStemMode() {
    // FULL_CASE_FOLD would lowercase a token-sourced term; in stem mode the
    // raw surface form feeds the stemmer, and Porter is case-sensitive.
    final AnalyzeResponse response = stub.analyze(AnalyzeRequest.newBuilder()
        .setText("Running running")
        .setOptions(AnalysisOptions.newBuilder()
            .setStemmer(AnalysisOptions.Stemmer.STEMMER_PORTER)
            .setTermVectors(stemmed(TermVectorOptions.Mode.MODE_FULL)
                .addSteps(TermVectorOptions.NormalizerStep.NORMALIZER_STEP_FULL_CASE_FOLD)))
        .build());

    assertThat(response.getTermVectorsList())
        .extracting(v -> v.getTerm() + ":" + v.getFrequency())
        .containsExactlyInAnyOrder("Run:1", "run:1");
  }

  @Test
  void stemModeWithoutStemmerIsInvalidArgument() {
    assertThatThrownBy(() -> stub.analyze(AnalyzeRequest.newBuilder()
        .setText("running runs")
        .setOptions(AnalysisOptions.newBuilder()
            .setTermVectors(stemmed(TermVectorOptions.Mode.MODE_FULL)))
        .build()))
        .isInstanceOfSatisfying(StatusRuntimeException.class, e -> {
          assertThat(e.getStatus().getCode()).isEqualTo(Status.Code.INVALID_ARGUMENT);
          assertThat(e.getStatus().getDescription()).contains("stemmer");
        });
  }

  @Test
  void normalizedStemsAcceptTheNonOffsetAwareSteps() {
    // NORMALIZED_STEMS runs the step chain (plain, per token) before the
    // stemmer. NFKC cannot report an alignment; this proves the fold-then-stem
    // path takes the new steps: full-width "ＦＩＳＨＩＮＧ" folds to "FISHING",
    // case-folds to "fishing", and stems to "fish", grouping with "fishing".
    final AnalyzeResponse response = stub.analyze(AnalyzeRequest.newBuilder()
        .setText("ＦＩＳＨＩＮＧ fishing")
        .setOptions(AnalysisOptions.newBuilder()
            .setStemmer(AnalysisOptions.Stemmer.STEMMER_PORTER)
            .setTermVectors(TermVectorOptions.newBuilder()
                .setEnabled(true)
                .setSource(TermVectorOptions.Source.SOURCE_NORMALIZED_STEMS)
                .setMode(TermVectorOptions.Mode.MODE_FULL)
                .addSteps(TermVectorOptions.NormalizerStep.NORMALIZER_STEP_NFKC)
                .addSteps(
                    TermVectorOptions.NormalizerStep.NORMALIZER_STEP_FULL_CASE_FOLD)))
        .build());

    assertThat(response.getTermVectorsCount()).isEqualTo(1);
    final ai.pipestream.opennlp.analysis.v1.TermVector vector = response.getTermVectors(0);
    assertThat(vector.getTerm()).isEqualTo("fish");
    assertThat(vector.getFrequency()).isEqualTo(2);
    assertThat(vector.getOccurrencesList())
        .extracting(s -> s.getStart() + "-" + s.getEnd())
        .containsExactly("0-7", "8-15");
  }
}
