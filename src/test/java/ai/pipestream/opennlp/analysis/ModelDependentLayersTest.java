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
import ai.pipestream.opennlp.analysis.v1.RelationOptions;
import io.grpc.ManagedChannel;
import io.grpc.Server;
import io.grpc.Status;
import io.grpc.StatusRuntimeException;
import io.grpc.inprocess.InProcessChannelBuilder;
import io.grpc.inprocess.InProcessServerBuilder;

/**
 * The model-dependent layers (coref, dependency parse, relations, geo)
 * against a server with NO models configured: every request still succeeds,
 * the layer stays empty, and a loud warning names the missing configuration.
 * The model-backed happy paths are covered by {@link ModelBackedLayersTest},
 * which trains tiny models in memory (model-file loading is broken in this
 * preview build).
 */
class ModelDependentLayersTest {

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

  private AnalyzeResponse analyze(AnalysisOptions.Builder options) {
    return stub.analyze(AnalyzeRequest.newBuilder()
        .setText("The court held that the defendant was liable.")
        .setOptions(options)
        .build());
  }

  @Test
  void corefWithoutModelsWarnsAndStaysEmpty() {
    final AnalyzeResponse response =
        analyze(AnalysisOptions.newBuilder().setCoref(true));

    assertThat(response.getCorefMentionsList()).isEmpty();
    assertThat(response.getWarningsList())
        .anySatisfy(w -> assertThat(w).contains("OPENNLP_POS_MODEL"));
  }

  @Test
  void dependencyParseWithoutModelWarnsAndStaysEmpty() {
    final AnalyzeResponse response =
        analyze(AnalysisOptions.newBuilder().setDependencyParse(true));

    assertThat(response.getDependenciesList()).isEmpty();
    assertThat(response.getWarningsList())
        .anySatisfy(w -> assertThat(w).contains("OPENNLP_DEPPARSE_MODEL"));
  }

  @Test
  void relationsWithoutModelsWarnAndStayEmpty() {
    final AnalyzeResponse response = analyze(AnalysisOptions.newBuilder()
        .setRelations(RelationOptions.newBuilder()
            .addPatterns(RelationOptions.Pattern.newBuilder()
                .setType("party-of")
                .setPath("<nsubj >obj"))));

    assertThat(response.getRelationsList()).isEmpty();
    // The dependency prerequisite fails first, and its warning names both
    // layers that stay empty.
    assertThat(response.getWarningsList())
        .anySatisfy(w -> assertThat(w).contains("OPENNLP_DEPPARSE_MODEL"));
  }

  @Test
  void relationsWithEmptyPatternsIsRejected() {
    assertThatThrownBy(() ->
        analyze(AnalysisOptions.newBuilder().setRelations(RelationOptions.newBuilder())))
        .isInstanceOfSatisfying(StatusRuntimeException.class, e ->
            assertThat(e.getStatus().getCode()).isEqualTo(Status.Code.INVALID_ARGUMENT));
  }

  @Test
  void geoWithoutNerModelWarnsAndStaysEmpty() {
    final AnalyzeResponse response = analyze(AnalysisOptions.newBuilder().setGeo(true));

    assertThat(response.getLocationsList()).isEmpty();
    assertThat(response.getRegionsList()).isEmpty();
    assertThat(response.getWarningsList())
        .anySatisfy(w -> assertThat(w).contains("OPENNLP_NER_MODEL"));
  }

  @Test
  void capabilitiesReportMissingModels() {
    final var capabilities = stub.getCapabilities(GetCapabilitiesRequest.getDefaultInstance());

    assertThat(capabilities.getDependencyParseAvailable()).isFalse();
    assertThat(capabilities.getNerAvailable()).isFalse();
    assertThat(capabilities.getPosTagsAvailable()).isFalse();
  }
}
