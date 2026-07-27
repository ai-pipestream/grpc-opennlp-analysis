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

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import ai.pipestream.opennlp.analysis.config.ServiceConfig;
import ai.pipestream.opennlp.analysis.pipeline.PipelineEnvironment;
import ai.pipestream.opennlp.analysis.v1.AnalysisOptions;
import ai.pipestream.opennlp.analysis.v1.AnalysisServiceGrpc;
import ai.pipestream.opennlp.analysis.v1.AnalyzeRequest;
import ai.pipestream.opennlp.analysis.v1.AnalyzeResponse;
import ai.pipestream.opennlp.analysis.v1.GetCapabilitiesRequest;
import ai.pipestream.opennlp.analysis.v1.GetCapabilitiesResponse;
import ai.pipestream.opennlp.analysis.v1.TermVectorOptions;
import io.grpc.ManagedChannel;
import io.grpc.Server;
import io.grpc.Status;
import io.grpc.StatusRuntimeException;
import io.grpc.health.v1.HealthCheckRequest;
import io.grpc.health.v1.HealthCheckResponse;
import io.grpc.health.v1.HealthGrpc;
import io.grpc.netty.shaded.io.grpc.netty.NettyServerBuilder;
import io.grpc.protobuf.services.HealthStatusManager;
import io.grpc.protobuf.services.ProtoReflectionServiceV1;

/**
 * Integration test over a real Netty server bound to a random loopback port
 * and a real Netty channel — the same wiring {@link AnalysisServer} uses.
 */
class AnalysisServerIntegrationTest {

  private static Server server;
  private static ManagedChannel channel;
  private static AnalysisServiceGrpc.AnalysisServiceBlockingStub stub;

  @BeforeAll
  static void setUp() throws Exception {
    final ServiceConfig config = new ServiceConfig(0, 1024 * 1024, null, null, null);
    final HealthStatusManager health = new HealthStatusManager();
    server = NettyServerBuilder.forPort(0)
        .addService(new AnalysisServiceImpl(PipelineEnvironment.load(config), config))
        .addService(health.getHealthService())
        .addService(ProtoReflectionServiceV1.newInstance())
        .build()
        .start();
    health.setStatus("", HealthCheckResponse.ServingStatus.SERVING);
    health.setStatus("ai.pipestream.opennlp.analysis.v1.AnalysisService",
        HealthCheckResponse.ServingStatus.SERVING);
    channel = io.grpc.netty.shaded.io.grpc.netty.NettyChannelBuilder
        .forAddress("127.0.0.1", server.getPort())
        .usePlaintext()
        .build();
    stub = AnalysisServiceGrpc.newBlockingStub(channel);
  }

  @AfterAll
  static void tearDown() {
    channel.shutdownNow();
    server.shutdownNow();
  }

  @Test
  void fullAnalyzeOverRealChannel() {
    final AnalyzeResponse response = stub.analyze(AnalyzeRequest.newBuilder()
        .setText("Running dogs bark.\nGroße Hunde laufen.")
        .setOptions(AnalysisOptions.newBuilder()
            .setLanguage("en")
            .setTokenizer(AnalysisOptions.Tokenizer.TOKENIZER_SIMPLE)
            .setSentenceDetection(true)
            .setStemmer(AnalysisOptions.Stemmer.STEMMER_SNOWBALL_ENGLISH)
            .setTermVectors(TermVectorOptions.newBuilder()
                .setEnabled(true)
                .setMode(TermVectorOptions.Mode.MODE_FULL)
                .addRungs(TermVectorOptions.NormalizerRung.NORMALIZER_RUNG_STRIP_INVISIBLE)
                .addRungs(TermVectorOptions.NormalizerRung.NORMALIZER_RUNG_WHITESPACE)
                .addRungs(TermVectorOptions.NormalizerRung.NORMALIZER_RUNG_FULL_CASE_FOLD)))
        .build());

    assertThat(response.getSentencesCount()).isEqualTo(2);
    assertThat(response.getSentences(0).getStart()).isZero();
    assertThat(response.getSentences(1).getEnd())
        .isEqualTo("Running dogs bark.\nGroße Hunde laufen.".length());

    assertThat(response.getTokensList()).extracting(t -> t.getText())
        .containsExactly("Running", "dogs", "bark", ".", "Große", "Hunde",
            "laufen", ".");
    // Every token span covers exactly its text in the original input.
    final String text = "Running dogs bark.\nGroße Hunde laufen.";
    assertThat(response.getTokensList()).allSatisfy(t ->
        assertThat(text.substring(t.getSpan().getStart(), t.getSpan().getEnd()))
            .isEqualTo(t.getText()));

    assertThat(response.getStemsList()).hasSameSizeAs(response.getTokensList());
    // Snowball does not lowercase; "Running" stems to "Run".
    assertThat(response.getStemsList().get(0)).isEqualTo("Run");

    // FULL mode: every occurrence carries original-text offsets.
    assertThat(response.getTermVectorsList()).isNotEmpty();
    assertThat(response.getTermVectorsList()).allSatisfy(v -> {
      assertThat(v.getFrequency()).isPositive();
      assertThat(v.getOccurrencesCount()).isEqualTo(v.getFrequency());
    });
    assertThat(response.getWarningsList()).isEmpty();
  }

  @Test
  void capabilitiesReflectEmbeddingsDisabled() {
    final GetCapabilitiesResponse capabilities =
        stub.getCapabilities(GetCapabilitiesRequest.getDefaultInstance());

    assertThat(capabilities.getEmbeddingsEnabled()).isFalse();
    assertThat(capabilities.getOpennlpVersion()).isEqualTo("3.x-preview-SNAPSHOT");
    assertThat(capabilities.getWarningsList())
        .anySatisfy(w -> assertThat(w).contains("OPENNLP_EMBEDDINGS_DIR"));
  }

  @Test
  void invalidRequestFailsWithInvalidArgument() {
    assertThatThrownBy(() -> stub.analyze(AnalyzeRequest.getDefaultInstance()))
        .isInstanceOfSatisfying(StatusRuntimeException.class,
            e -> assertThat(e.getStatus().getCode())
                .isEqualTo(Status.Code.INVALID_ARGUMENT));
  }

  @Test
  void healthServiceReportsServing() {
    assertThat(HealthGrpc.newBlockingStub(channel)
        .check(HealthCheckRequest.getDefaultInstance()).getStatus())
        .isEqualTo(HealthCheckResponse.ServingStatus.SERVING);
    assertThat(HealthGrpc.newBlockingStub(channel)
        .check(HealthCheckRequest.newBuilder()
            .setService("ai.pipestream.opennlp.analysis.v1.AnalysisService")
            .build()).getStatus())
        .isEqualTo(HealthCheckResponse.ServingStatus.SERVING);
  }
}
