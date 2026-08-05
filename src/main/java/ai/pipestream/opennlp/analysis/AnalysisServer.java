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

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ai.pipestream.opennlp.analysis.config.ServiceConfig;
import ai.pipestream.opennlp.analysis.pipeline.PipelineEnvironment;
import ai.pipestream.opennlp.analysis.vocab.VocabularyListener;
import io.grpc.Server;
import io.grpc.health.v1.HealthCheckResponse;
import io.grpc.netty.shaded.io.grpc.netty.NettyServerBuilder;
import io.grpc.protobuf.services.HealthStatusManager;
import io.grpc.protobuf.services.ProtoReflectionServiceV1;

/**
 * Entry point of the OpenNLP analysis gRPC service.
 *
 * <p>Port resolution: first command line argument, then the {@code PORT}
 * environment variable, then {@link ServiceConfig#DEFAULT_PORT}. The server
 * exposes the analysis service, the gRPC health service, and server
 * reflection. Shutdown is graceful: {@code shutdown()}, up to five seconds for
 * in-flight calls, then {@code shutdownNow()}.</p>
 */
public final class AnalysisServer {

  private static final Logger LOG = LoggerFactory.getLogger(AnalysisServer.class);

  private static final int SHUTDOWN_GRACE_SECONDS = 5;

  private AnalysisServer() {
  }

  /**
   * Starts the server and blocks until it terminates.
   *
   * @param args {@code args[0]} is the listen port when present
   * @throws Exception when the server cannot start or is interrupted
   */
  public static void main(String[] args) throws Exception {
    final ServiceConfig config = ServiceConfig.fromEnvironment(args);
    final PipelineEnvironment environment = PipelineEnvironment.load(config);
    final HealthStatusManager health = new HealthStatusManager();

    // The vocabulary listener is absent unless OPENNLP_VOCAB_DIR is set. A
    // configured but unusable directory degrades to disabled with a loud
    // GetCapabilities warning — vocabulary statistics must never take the
    // analysis service down.
    final List<String> serverWarnings = new ArrayList<>();
    VocabularyListener vocabularyListener = null;
    if (config.vocabDir() != null) {
      try {
        vocabularyListener = new VocabularyListener(config.vocabDir(),
            config.vocabWindowDocs(), config.vocabTopK(), config.embeddingsDir());
      } catch (IOException | RuntimeException e) {
        final String warning = "OPENNLP_VOCAB_DIR is set to " + config.vocabDir()
            + " but the vocabulary listener could not start: " + e.getMessage()
            + "; vocabulary statistics are disabled, analysis is unaffected";
        LOG.error("Vocabulary listener disabled", e);
        serverWarnings.add(warning);
      }
    }

    final NettyServerBuilder builder = NettyServerBuilder.forPort(config.port())
        // The transport cap must track the configured text cap: Netty's
        // 4 MiB default would reject requests OPENNLP_MAX_TEXT_BYTES allows.
        .maxInboundMessageSize(Math.max(4 * 1024 * 1024, config.maxTextBytes() + 1024 * 1024))
        // Eager headers make call acceptance visible before the first
        // result, which AnalyzeStream clients await before submitting.
        .addService(io.grpc.ServerInterceptors.intercept(
            new AnalysisServiceImpl(environment, config, vocabularyListener,
                serverWarnings), new EagerHeadersInterceptor()))
        .addService(health.getHealthService())
        .addService(ProtoReflectionServiceV1.newInstance());
    if (vocabularyListener != null) {
      builder.addService(new VocabServiceImpl(vocabularyListener));
    }
    final Server server = builder.build().start();

    health.setStatus("", HealthCheckResponse.ServingStatus.SERVING);
    health.setStatus("ai.pipestream.opennlp.analysis.v1.AnalysisService",
        HealthCheckResponse.ServingStatus.SERVING);
    if (vocabularyListener != null) {
      health.setStatus("ai.pipestream.opennlp.analysis.v1.VocabularyService",
          HealthCheckResponse.ServingStatus.SERVING);
    }

    LOG.info("OpenNLP analysis service listening on port {} (opennlp {})",
        server.getPort(), Versions.opennlp());

    Runtime.getRuntime().addShutdownHook(new Thread(() -> {
      LOG.info("Shutdown requested");
      server.shutdown();
      try {
        if (!server.awaitTermination(SHUTDOWN_GRACE_SECONDS, TimeUnit.SECONDS)) {
          LOG.warn("Forcing shutdown after {} s grace period", SHUTDOWN_GRACE_SECONDS);
          server.shutdownNow();
        }
      } catch (InterruptedException e) {
        Thread.currentThread().interrupt();
        server.shutdownNow();
      }
    }, "analysis-server-shutdown"));

    server.awaitTermination();
  }
}
