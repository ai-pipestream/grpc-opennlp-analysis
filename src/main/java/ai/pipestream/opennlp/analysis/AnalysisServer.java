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

import java.util.concurrent.TimeUnit;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ai.pipestream.opennlp.analysis.config.ServiceConfig;
import ai.pipestream.opennlp.analysis.pipeline.PipelineEnvironment;
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

    final Server server = NettyServerBuilder.forPort(config.port())
        .addService(new AnalysisServiceImpl(environment, config))
        .addService(health.getHealthService())
        .addService(ProtoReflectionServiceV1.newInstance())
        .build()
        .start();

    health.setStatus("", HealthCheckResponse.ServingStatus.SERVING);
    health.setStatus("ai.pipestream.opennlp.analysis.v1.AnalysisService",
        HealthCheckResponse.ServingStatus.SERVING);

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
