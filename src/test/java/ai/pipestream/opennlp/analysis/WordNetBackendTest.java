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

import java.nio.file.Files;
import java.nio.file.Path;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import ai.pipestream.opennlp.analysis.config.ServiceConfig;
import ai.pipestream.opennlp.analysis.pipeline.PipelineEnvironment;
import ai.pipestream.opennlp.analysis.pipeline.WordNetLemmatizer;
import ai.pipestream.opennlp.analysis.v1.AnalysisOptions;
import ai.pipestream.opennlp.analysis.v1.AnalysisServiceGrpc;
import ai.pipestream.opennlp.analysis.v1.AnalyzeRequest;
import ai.pipestream.opennlp.analysis.v1.AnalyzeResponse;
import ai.pipestream.opennlp.analysis.v1.GetCapabilitiesRequest;
import io.grpc.ManagedChannel;
import io.grpc.Server;
import io.grpc.inprocess.InProcessChannelBuilder;
import io.grpc.inprocess.InProcessServerBuilder;

/**
 * The WordNet lemmatizer backend end to end: environment loading from
 * OPENNLP_WORDNET_DIR, precedence over the flat dictionary, and the wire
 * lemmatize path over the mini WordNet fixture.
 */
class WordNetBackendTest {

  private Server server;
  private ManagedChannel channel;
  private AnalysisServiceGrpc.AnalysisServiceBlockingStub stub;

  private static Path wordnetDir() throws Exception {
    return Path.of(WordNetBackendTest.class
        .getResource("/opennlp/wordnet/mini-wndb").toURI());
  }

  private static ServiceConfig configWithWordNet(Path wordnetDir) {
    return new ServiceConfig(0, 1024 * 1024, null, null, null, null, null, null,
        null, null, null, wordnetDir, 0);
  }

  @BeforeEach
  void setUp() throws Exception {
    final String name = InProcessServerBuilder.generateName();
    server = InProcessServerBuilder.forName(name).directExecutor()
        .addService(new AnalysisServiceImpl(
            PipelineEnvironment.load(configWithWordNet(wordnetDir())),
            configWithWordNet(wordnetDir())))
        .build().start();
    channel = InProcessChannelBuilder.forName(name).directExecutor().build();
    stub = AnalysisServiceGrpc.newBlockingStub(channel);
  }

  @AfterEach
  void tearDown() {
    channel.shutdownNow();
    server.shutdownNow();
  }

  @Test
  void environmentLoadsTheWordNetBackend() throws Exception {
    final PipelineEnvironment environment =
        PipelineEnvironment.load(configWithWordNet(wordnetDir()));

    assertThat(environment.lemmatizer()).isInstanceOf(WordNetLemmatizer.class);
    assertThat(environment.loadWarnings()).noneMatch(w -> w.contains("WORDNET"));
  }

  @Test
  void wordNetWinsOverTheFlatDictionaryWithAWarning(@TempDir Path tempDir)
      throws Exception {
    final Path flatDict = tempDir.resolve("lemmas.tsv");
    Files.writeString(flatDict, "dogs\t\tdog\n");
    final ServiceConfig both = new ServiceConfig(0, 1024 * 1024, null, null, null,
        null, null, flatDict, null, null, null, wordnetDir(), 0);

    final PipelineEnvironment environment = PipelineEnvironment.load(both);

    assertThat(environment.lemmatizer()).isInstanceOf(WordNetLemmatizer.class);
    assertThat(environment.loadWarnings())
        .anySatisfy(w -> assertThat(w)
            .contains("OPENNLP_WORDNET_DIR").contains("OPENNLP_LEMMATIZER_DICT"));
  }

  @Test
  void lemmatizeOverTheWireUsesWordNet() {
    final AnalyzeResponse response = stub.analyze(AnalyzeRequest.newBuilder()
        .setText("running dogs men")
        .setOptions(AnalysisOptions.newBuilder().setLemmatize(true))
        .build());

    // No POS model is configured, so every token takes the all-POS fallback:
    // the verb exception, the noun rule, and the noun exception.
    assertThat(response.getLemmasList()).containsExactly("run", "dog", "man");
    assertThat(response.getWarningsList()).isEmpty();
  }

  @Test
  void capabilitiesReportTheWordNetBackendAsLemmatizer() {
    assertThat(stub.getCapabilities(GetCapabilitiesRequest.getDefaultInstance())
        .getLemmatizerAvailable()).isTrue();
  }
}
