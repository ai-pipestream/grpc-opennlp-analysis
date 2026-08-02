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
import ai.pipestream.opennlp.analysis.v1.Token;
import io.grpc.ManagedChannel;
import io.grpc.Server;
import io.grpc.inprocess.InProcessChannelBuilder;
import io.grpc.inprocess.InProcessServerBuilder;

/**
 * Tokenizer selection over the wire: UAX29 is model-free and always served;
 * the lattice and SentencePiece tokenizers fall back to whitespace with a
 * loud warning when the server has no dictionary/model configured.
 */
class TokenizerSelectionTest {

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

  private AnalyzeResponse analyze(String text, AnalysisOptions.Tokenizer tokenizer) {
    return stub.analyze(AnalyzeRequest.newBuilder()
        .setText(text)
        .setOptions(AnalysisOptions.newBuilder().setTokenizer(tokenizer))
        .build());
  }

  @Test
  void uax29DropsPunctuationAndKeepsAbbreviationPeriods() {
    final AnalyzeResponse response =
        analyze("Hello, world!", AnalysisOptions.Tokenizer.TOKENIZER_UAX29);

    assertThat(response.getTokensList())
        .extracting(Token::getText)
        .containsExactly("Hello", "world");
    assertThat(response.getTokensList().get(0).getSpan().getStart()).isEqualTo(0);
    assertThat(response.getTokensList().get(0).getSpan().getEnd()).isEqualTo(5);
    assertThat(response.getTokensList().get(1).getSpan().getStart()).isEqualTo(7);
    assertThat(response.getTokensList().get(1).getSpan().getEnd()).isEqualTo(12);
    assertThat(response.getWarningsList()).isEmpty();
  }

  @Test
  void uax29KeepsCitationShapesTogether() {
    final AnalyzeResponse response = analyze("Roe v. Wade, 410 U.S. 113",
        AnalysisOptions.Tokenizer.TOKENIZER_UAX29);

    // Whitespace tokenization would keep "Wade,"; UAX29 splits the comma off
    // and keeps the "U.S" abbreviation as one token.
    assertThat(response.getTokensList())
        .extracting(Token::getText)
        .containsExactly("Roe", "v", "Wade", "410", "U.S", "113");
  }

  @Test
  void latticeWithoutDictionaryFallsBackToWhitespaceWithAWarning() {
    final AnalyzeResponse response =
        analyze("some text", AnalysisOptions.Tokenizer.TOKENIZER_LATTICE);

    assertThat(response.getTokensList())
        .extracting(Token::getText)
        .containsExactly("some", "text");
    assertThat(response.getWarningsList())
        .singleElement()
        .satisfies(w -> assertThat(w).contains("OPENNLP_LATTICE_DIC_DIR"));
  }

  @Test
  void sentencepieceWithoutModelFallsBackToWhitespaceWithAWarning() {
    final AnalyzeResponse response =
        analyze("some text", AnalysisOptions.Tokenizer.TOKENIZER_SENTENCEPIECE);

    assertThat(response.getTokensList())
        .extracting(Token::getText)
        .containsExactly("some", "text");
    assertThat(response.getWarningsList())
        .singleElement()
        .satisfies(w -> assertThat(w).contains("OPENNLP_SENTENCEPIECE_MODEL"));
  }

  @Test
  void capabilitiesReportTokenizerAvailability() {
    final var capabilities = stub.getCapabilities(GetCapabilitiesRequest.getDefaultInstance());

    assertThat(capabilities.getTokenizersList())
        .contains("TOKENIZER_UAX29", "TOKENIZER_LATTICE", "TOKENIZER_SENTENCEPIECE");
    assertThat(capabilities.getLatticeAvailable()).isFalse();
    assertThat(capabilities.getSentencepieceAvailable()).isFalse();
  }
}
