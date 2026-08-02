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
import ai.pipestream.opennlp.analysis.v1.PiiMention;
import io.grpc.ManagedChannel;
import io.grpc.Server;
import io.grpc.inprocess.InProcessChannelBuilder;
import io.grpc.inprocess.InProcessServerBuilder;

/**
 * PII extraction over the wire: model-free, always available, spans in
 * original-text coordinates, normalized forms attached.
 */
class PiiExtractionTest {

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

  private AnalyzeResponse analyze(String text, boolean pii) {
    return stub.analyze(AnalyzeRequest.newBuilder()
        .setText(text)
        .setOptions(AnalysisOptions.newBuilder().setPii(pii))
        .build());
  }

  @Test
  void extractsEmailAndPhoneWithOriginalSpans() {
    final AnalyzeResponse response = analyze(
        "reach me at john.doe@example.com or call +1-202-555-0143", true);

    assertThat(response.getPiiList())
        .extracting(PiiMention::getType)
        .containsExactly("email", "phone");
    final PiiMention email = response.getPiiList().get(0);
    assertThat(email.getSpan().getStart()).isEqualTo(12);
    assertThat(email.getSpan().getEnd()).isEqualTo(32);
    assertThat(email.getNormalized()).isEqualTo("john.doe@example.com");
    final PiiMention phone = response.getPiiList().get(1);
    assertThat(phone.getSpan().getStart()).isEqualTo(41);
    assertThat(phone.getSpan().getEnd()).isEqualTo(56);
    // The normalized form strips the phone's punctuation but keeps the '+'.
    assertThat(phone.getNormalized()).isEqualTo("+12025550143");
    assertThat(response.getWarningsList()).isEmpty();
  }

  @Test
  void extractsIban() {
    final AnalyzeResponse response = analyze("IBAN DE89370400440532013000 thanks", true);

    assertThat(response.getPiiList()).singleElement().satisfies(mention -> {
      assertThat(mention.getType()).isEqualTo("iban");
      assertThat(mention.getSpan().getStart()).isEqualTo(5);
      assertThat(mention.getSpan().getEnd()).isEqualTo(27);
    });
  }

  @Test
  void cleanTextProducesNoMentions() {
    final AnalyzeResponse response = analyze("plain text nothing here", true);

    assertThat(response.getPiiList()).isEmpty();
    assertThat(response.getWarningsList()).isEmpty();
  }

  @Test
  void layerStaysEmptyUnlessRequested() {
    final AnalyzeResponse response = analyze("reach john.doe@example.com today", false);

    assertThat(response.getPiiList()).isEmpty();
  }
}
