/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License. You may obtain a copy of the License at
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
import ai.pipestream.opennlp.analysis.v1.GetCapabilitiesRequest;
import ai.pipestream.opennlp.analysis.v1.TermVectorOptions;
import io.grpc.ManagedChannel;
import io.grpc.Server;
import io.grpc.Status;
import io.grpc.StatusRuntimeException;
import io.grpc.inprocess.InProcessChannelBuilder;
import io.grpc.inprocess.InProcessServerBuilder;

/**
 * Dual term identity: one analysis pass returns the folded term vectors and
 * the cased term vectors, both computed over the same tokens, so their
 * occurrence spans coincide. The cased chain is the requested chain minus the
 * case-folding steps.
 */
class DualTermIdentityTest {

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

  private static TermVectorOptions.Builder dual(TermVectorOptions.Source source) {
    return TermVectorOptions.newBuilder()
        .setEnabled(true)
        .setSource(source)
        .setMode(TermVectorOptions.Mode.MODE_FULL)
        .addSteps(TermVectorOptions.NormalizerStep.NORMALIZER_STEP_FULL_CASE_FOLD)
        .setDualCased(true);
  }

  @Test
  void onePassYieldsFoldedAndCasedIdentitiesWithSharedSpans() {
    final AnalyzeResponse response = stub.analyze(AnalyzeRequest.newBuilder()
        .setText("COURT court Court Dungeons dungeons running runs")
        .setOptions(AnalysisOptions.newBuilder()
            .setStemmer(AnalysisOptions.Stemmer.STEMMER_PORTER)
            .setTermVectors(dual(TermVectorOptions.Source.SOURCE_NORMALIZED_STEMS)))
        .build());

    // Folded: case folded before stemming, so every casing variant collapses.
    assertThat(response.getTermVectorsList())
        .extracting(v -> v.getTerm() + ":" + v.getFrequency())
        .containsExactlyInAnyOrder("court:3", "dungeon:2", "run:2");

    // Cased: the same chain minus the case fold, so case variants stay apart
    // while inflection still groups ("Dungeons" -> "Dungeon", "runs" -> "run").
    assertThat(response.getCasedTermVectorsList())
        .extracting(v -> v.getTerm() + ":" + v.getFrequency())
        .containsExactlyInAnyOrder(
            "COURT:1", "court:1", "Court:1", "Dungeon:1", "dungeon:1", "run:2");

    // Both identities anchor the same occurrences: the folded "court" covers
    // all three casings, the cased lists partition the same spans, and the
    // cased "run" carries the running/runs spans.
    assertThat(response.getTermVectorsList().stream()
        .filter(v -> v.getTerm().equals("court"))
        .flatMap(v -> v.getOccurrencesList().stream())
        .map(s -> s.getStart() + "-" + s.getEnd()))
        .containsExactly("0-5", "6-11", "12-17");
    assertThat(response.getCasedTermVectorsList().stream()
        .filter(v -> v.getTerm().equals("Court"))
        .flatMap(v -> v.getOccurrencesList().stream())
        .map(s -> s.getStart() + "-" + s.getEnd()))
        .containsExactly("12-17");
    assertThat(response.getCasedTermVectorsList().stream()
        .filter(v -> v.getTerm().equals("run"))
        .flatMap(v -> v.getOccurrencesList().stream())
        .map(s -> s.getStart() + "-" + s.getEnd()))
        .containsExactly("36-43", "44-48");
  }

  @Test
  void tokenSourcedDualIdentityKeepsCaseApart() {
    final AnalyzeResponse response = stub.analyze(AnalyzeRequest.newBuilder()
        .setText("Dog dogs DOG")
        .setOptions(AnalysisOptions.newBuilder()
            .setTermVectors(dual(TermVectorOptions.Source.SOURCE_TOKENS)))
        .build());

    assertThat(response.getTermVectorsList())
        .extracting(v -> v.getTerm() + ":" + v.getFrequency())
        .containsExactlyInAnyOrder("dog:2", "dogs:1");
    assertThat(response.getCasedTermVectorsList())
        .extracting(v -> v.getTerm() + ":" + v.getFrequency())
        .containsExactlyInAnyOrder("Dog:1", "dogs:1", "DOG:1");
  }

  @Test
  void dualCasedWithoutCaseFoldStepWarnsAndDuplicates() {
    final AnalyzeResponse response = stub.analyze(AnalyzeRequest.newBuilder()
        .setText("Dog dogs")
        .setOptions(AnalysisOptions.newBuilder()
            .setTermVectors(TermVectorOptions.newBuilder()
                .setEnabled(true)
                .setSource(TermVectorOptions.Source.SOURCE_TOKENS)
                .setMode(TermVectorOptions.Mode.MODE_FULL)
                .addSteps(TermVectorOptions.NormalizerStep.NORMALIZER_STEP_WHITESPACE)
                .setDualCased(true)))
        .build());

    assertThat(response.getWarningsList())
        .anySatisfy(w -> assertThat(w).contains("no case-folding step"));
    assertThat(response.getCasedTermVectorsList())
        .extracting(v -> v.getTerm() + ":" + v.getFrequency())
        .containsExactlyInAnyOrder("Dog:1", "dogs:1");
    assertThat(response.getTermVectorsList())
        .extracting(v -> v.getTerm() + ":" + v.getFrequency())
        .containsExactlyInAnyOrder("Dog:1", "dogs:1");
  }

  @Test
  void dualCasedWithPlainStemSourceIsInvalidArgument() {
    assertThatThrownBy(() -> stub.analyze(AnalyzeRequest.newBuilder()
        .setText("running runs")
        .setOptions(AnalysisOptions.newBuilder()
            .setStemmer(AnalysisOptions.Stemmer.STEMMER_PORTER)
            .setTermVectors(dual(TermVectorOptions.Source.SOURCE_STEMS)))
        .build()))
        .isInstanceOfSatisfying(StatusRuntimeException.class, e -> {
          assertThat(e.getStatus().getCode()).isEqualTo(Status.Code.INVALID_ARGUMENT);
          assertThat(e.getStatus().getDescription()).contains("dual_cased");
        });
  }

  @Test
  void capabilitiesAdvertiseDualTermIdentity() {
    assertThat(stub.getCapabilities(GetCapabilitiesRequest.getDefaultInstance())
        .getDualTermIdentityAvailable()).isTrue();
  }
}
