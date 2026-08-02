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
import ai.pipestream.opennlp.analysis.v1.GetCapabilitiesRequest;
import ai.pipestream.opennlp.analysis.v1.TermVectorOptions;
import io.grpc.ManagedChannel;
import io.grpc.Server;
import io.grpc.Status;
import io.grpc.StatusRuntimeException;
import io.grpc.inprocess.InProcessChannelBuilder;
import io.grpc.inprocess.InProcessServerBuilder;

/**
 * The de-hyphenation step: the one count-changing normalizer step. A
 * line-break hyphenation ("litiga-\ntion") joins into a single term whose
 * occurrence span covers the hyphen and the break in the ORIGINAL text, and
 * the step is rejected the moment the chain or the term source cannot honor
 * that mapping.
 */
class DehyphenationStepTest {

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

  private static TermVectorOptions.Builder tokens(TermVectorOptions.NormalizerStep... steps) {
    final TermVectorOptions.Builder termVectors = TermVectorOptions.newBuilder()
        .setEnabled(true)
        .setSource(TermVectorOptions.Source.SOURCE_TOKENS)
        .setMode(TermVectorOptions.Mode.MODE_FULL);
    for (final TermVectorOptions.NormalizerStep step : steps) {
      termVectors.addSteps(step);
    }
    return termVectors;
  }

  @Test
  void lineBreakHyphenationJoinsIntoOneTermWithTheOriginalSpan() {
    final String broken = "litiga-\ntion";
    final String text = "the " + broken + " continued";
    final int start = text.indexOf(broken);
    final int end = start + broken.length();

    final AnalyzeResponse response = stub.analyze(AnalyzeRequest.newBuilder()
        .setText(text)
        .setOptions(AnalysisOptions.newBuilder()
            .setTermVectors(tokens(
                TermVectorOptions.NormalizerStep.NORMALIZER_STEP_DEHYPHENATE,
                TermVectorOptions.NormalizerStep.NORMALIZER_STEP_FULL_CASE_FOLD)))
        .build());

    // The count-changing proof: the token layer still holds the two broken
    // halves, but they group under ONE term entry.
    assertThat(response.getTokensList())
        .extracting(t -> t.getText())
        .containsExactly("the", "litiga-", "tion", "continued");
    assertThat(response.getTermVectorsList())
        .extracting(v -> v.getTerm() + ":" + v.getFrequency())
        .containsExactlyInAnyOrder("the:1", "litigation:1", "continued:1");

    // The joined term's span is the exact original range covering both halves
    // and the deleted hyphen and line break between them.
    assertThat(response.getTermVectorsList().stream()
        .filter(v -> v.getTerm().equals("litigation"))
        .flatMap(v -> v.getOccurrencesList().stream())
        .map(s -> s.getStart() + "-" + s.getEnd()))
        .containsExactly(start + "-" + end);
    assertThat(text.substring(start, end)).isEqualTo(broken);
    assertThat(response.getWarningsList()).isEmpty();
  }

  @Test
  void dehyphenateWithANonOffsetAwareStepIsInvalidArgument() {
    assertThatThrownBy(() -> stub.analyze(AnalyzeRequest.newBuilder()
        .setText("litiga-\ntion")
        .setOptions(AnalysisOptions.newBuilder()
            .setTermVectors(tokens(
                TermVectorOptions.NormalizerStep.NORMALIZER_STEP_DEHYPHENATE,
                TermVectorOptions.NormalizerStep.NORMALIZER_STEP_NFKC)))
        .build()))
        .isInstanceOfSatisfying(StatusRuntimeException.class, e -> {
          assertThat(e.getStatus().getCode()).isEqualTo(Status.Code.INVALID_ARGUMENT);
          assertThat(e.getStatus().getDescription())
              .contains("NORMALIZER_STEP_DEHYPHENATE").contains("offset-aware");
        });
  }

  @Test
  void dehyphenateWithStemSourceIsInvalidArgument() {
    assertThatThrownBy(() -> stub.analyze(AnalyzeRequest.newBuilder()
        .setText("litiga-\ntion")
        .setOptions(AnalysisOptions.newBuilder()
            .setStemmer(AnalysisOptions.Stemmer.STEMMER_PORTER)
            .setTermVectors(TermVectorOptions.newBuilder()
                .setEnabled(true)
                .setSource(TermVectorOptions.Source.SOURCE_STEMS)
                .setMode(TermVectorOptions.Mode.MODE_FULL)
                .addSteps(TermVectorOptions.NormalizerStep.NORMALIZER_STEP_DEHYPHENATE)))
        .build()))
        .isInstanceOfSatisfying(StatusRuntimeException.class, e -> {
          assertThat(e.getStatus().getCode()).isEqualTo(Status.Code.INVALID_ARGUMENT);
          assertThat(e.getStatus().getDescription())
              .contains("NORMALIZER_STEP_DEHYPHENATE").contains("SOURCE_TOKENS");
        });
  }

  @Test
  void capabilitiesAdvertiseDehyphenation() {
    assertThat(stub.getCapabilities(GetCapabilitiesRequest.getDefaultInstance())
        .getDehyphenationAvailable()).isTrue();
  }
}
