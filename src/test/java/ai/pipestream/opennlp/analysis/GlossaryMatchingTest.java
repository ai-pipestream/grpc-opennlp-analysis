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
import ai.pipestream.opennlp.analysis.v1.GlossaryOptions;
import io.grpc.ManagedChannel;
import io.grpc.Server;
import io.grpc.StatusRuntimeException;
import io.grpc.inprocess.InProcessChannelBuilder;
import io.grpc.inprocess.InProcessServerBuilder;

/**
 * Glossary matching over the wire: multi-word phrases match as one unit with
 * original-text spans, word boundaries are respected, and case folding is
 * optional.
 */
class GlossaryMatchingTest {

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

  private static GlossaryOptions.Builder glossary() {
    return GlossaryOptions.newBuilder()
        .addEntries(GlossaryOptions.Entry.newBuilder()
            .setId("game:dnd").setTerm("Dungeons & Dragons"))
        .addEntries(GlossaryOptions.Entry.newBuilder()
            .setId("case:roe").setTerm("Roe v. Wade"));
  }

  private AnalyzeResponse analyze(String text, GlossaryOptions.Builder glossary) {
    return stub.analyze(AnalyzeRequest.newBuilder()
        .setText(text)
        .setOptions(AnalysisOptions.newBuilder().setGlossary(glossary))
        .build());
  }

  @Test
  void multiWordPhrasesMatchAsOneUnit() {
    final AnalyzeResponse response = analyze(
        "The Dungeons & Dragons campaign cites Roe v. Wade often.", glossary());

    assertThat(response.getGlossaryMatchesList()).hasSize(2);
    assertThat(response.getGlossaryMatchesList().get(0)).satisfies(match -> {
      assertThat(match.getId()).isEqualTo("game:dnd");
      assertThat(match.getTerm()).isEqualTo("Dungeons & Dragons");
      assertThat(match.getSpan().getStart()).isEqualTo(4);
      assertThat(match.getSpan().getEnd()).isEqualTo(22);
    });
    assertThat(response.getGlossaryMatchesList().get(1)).satisfies(match -> {
      assertThat(match.getId()).isEqualTo("case:roe");
      assertThat(match.getSpan().getStart()).isEqualTo(38);
      assertThat(match.getSpan().getEnd()).isEqualTo(49);
    });
    assertThat(response.getWarningsList()).isEmpty();
  }

  @Test
  void matchingIsCaseSensitiveByDefaultAndFoldableOnRequest() {
    final AnalyzeResponse sensitive = analyze("a dungeons & dragons game", glossary());
    assertThat(sensitive.getGlossaryMatchesList()).isEmpty();

    final AnalyzeResponse folded =
        analyze("a dungeons & dragons game", glossary().setIgnoreCase(true));
    assertThat(folded.getGlossaryMatchesList()).singleElement().satisfies(match -> {
      assertThat(match.getId()).isEqualTo("game:dnd");
      // The registered term is reported even though the text differs in case.
      assertThat(match.getTerm()).isEqualTo("Dungeons & Dragons");
      assertThat(match.getSpan().getStart()).isEqualTo(2);
      assertThat(match.getSpan().getEnd()).isEqualTo(20);
    });
  }

  @Test
  void wordBoundariesBlockPartialMatches() {
    final AnalyzeResponse response = analyze("Dungeons & Dragon slayers", glossary());

    // "Dungeons & Dragons" does not match inside "Dungeons & Dragon slayers":
    // the letter after "Dragon" is a word character, blocking the hit.
    assertThat(response.getGlossaryMatchesList()).isEmpty();
  }

  @Test
  void overlappingMatchesResolveLeftmostLongest() {
    final AnalyzeResponse response = stub.analyze(AnalyzeRequest.newBuilder()
        .setText("Roe v. Wade")
        .setOptions(AnalysisOptions.newBuilder()
            .setGlossary(GlossaryOptions.newBuilder()
                .addEntries(GlossaryOptions.Entry.newBuilder()
                    .setId("short").setTerm("Roe v."))
                .addEntries(GlossaryOptions.Entry.newBuilder()
                    .setId("long").setTerm("Roe v. Wade"))))
        .build());

    assertThat(response.getGlossaryMatchesList()).singleElement()
        .satisfies(match -> assertThat(match.getId()).isEqualTo("long"));
  }

  @Test
  void emptyGlossaryIsRejected() {
    assertThatThrownBy(() -> analyze("text", GlossaryOptions.newBuilder()))
        .isInstanceOfSatisfying(StatusRuntimeException.class, e ->
            assertThat(e.getStatus().getCode())
                .isEqualTo(io.grpc.Status.Code.INVALID_ARGUMENT));
  }

  @Test
  void matchesStayEmptyWithoutAGlossary() {
    final AnalyzeResponse response = stub.analyze(AnalyzeRequest.newBuilder()
        .setText("The Dungeons & Dragons campaign.")
        .setOptions(AnalysisOptions.newBuilder().build())
        .build());

    assertThat(response.getGlossaryMatchesList()).isEmpty();
  }
}
