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
import opennlp.tools.lemmatizer.DictionaryLemmatizer;

/**
 * The morfologik lemmatizer backend end to end: environment loading from
 * OPENNLP_MORFOLOGIK_DICT, precedence between the three backends, the missing
 * file and missing metadata rules, and the wire lemmatize path over the mini
 * CFSA2 fixture (cat/cats/dogs/mice, SUFFIX encoder, '+' separator,
 * ISO-8859-1, neutral empty tags — the tags the service looks up when no POS
 * model produced any).
 */
class MorfologikBackendTest {

  private Server server;
  private ManagedChannel channel;
  private AnalysisServiceGrpc.AnalysisServiceBlockingStub stub;

  private static Path morfologikDict() throws Exception {
    return Path.of(MorfologikBackendTest.class
        .getResource("/opennlp/morfologik/mini-english.dict").toURI());
  }

  private static Path wordnetDir() throws Exception {
    return Path.of(MorfologikBackendTest.class
        .getResource("/opennlp/wordnet/mini-wndb").toURI());
  }

  private static ServiceConfig configWithMorfologik(Path dict) {
    return new ServiceConfig(0, 1024 * 1024, null, null, null, null, null, null,
        null, null, null, null, dict, null, 0);
  }

  @BeforeEach
  void setUp() throws Exception {
    final String name = InProcessServerBuilder.generateName();
    server = InProcessServerBuilder.forName(name).directExecutor()
        .addService(new AnalysisServiceImpl(
            PipelineEnvironment.load(configWithMorfologik(morfologikDict())),
            configWithMorfologik(morfologikDict())))
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
  void environmentLoadsTheMorfologikBackend() throws Exception {
    final PipelineEnvironment environment =
        PipelineEnvironment.load(configWithMorfologik(morfologikDict()));

    assertThat(environment.lemmatizer()).isInstanceOf(DictionaryLemmatizer.class);
    assertThat(environment.loadWarnings()).noneMatch(w -> w.contains("MORFOLOGIK"));
  }

  @Test
  void lemmatizeOverTheWireUsesMorfologik() {
    final AnalyzeResponse response = stub.analyze(AnalyzeRequest.newBuilder()
        .setText("cats dogs mice zzz")
        .setOptions(AnalysisOptions.newBuilder().setLemmatize(true))
        .build());

    // No POS model is configured, so every token looks up with the neutral
    // empty tag, which is exactly how the fixture's entries are keyed;
    // unknown vocabulary comes back as "O", the OpenNLP convention.
    assertThat(response.getLemmasList()).containsExactly("cat", "dog", "mouse", "O");
    assertThat(response.getWarningsList()).isEmpty();
  }

  @Test
  void wordNetWinsOverMorfologikWithAWarning() throws Exception {
    final ServiceConfig both = new ServiceConfig(0, 1024 * 1024, null, null, null,
        null, null, null, null, null, null, wordnetDir(), morfologikDict(), null, 0);

    final PipelineEnvironment environment = PipelineEnvironment.load(both);

    assertThat(environment.lemmatizer()).isInstanceOf(WordNetLemmatizer.class);
    assertThat(environment.loadWarnings())
        .anySatisfy(w -> assertThat(w)
            .contains("OPENNLP_WORDNET_DIR").contains("OPENNLP_MORFOLOGIK_DICT"));
  }

  @Test
  void morfologikWinsOverTheFlatDictionaryWithAWarning(@TempDir Path tempDir)
      throws Exception {
    final Path flatDict = tempDir.resolve("lemmas.tsv");
    Files.writeString(flatDict, "dogs\t\tdog\n");
    final ServiceConfig both = new ServiceConfig(0, 1024 * 1024, null, null, null,
        null, null, flatDict, null, null, null, null, morfologikDict(), null, 0);

    final PipelineEnvironment environment = PipelineEnvironment.load(both);

    // The flat dictionary knows only "dogs"; a "cats" hit proves the
    // morfologik backend serves.
    assertThat(environment.lemmatizer()).isInstanceOf(DictionaryLemmatizer.class);
    assertThat(environment.lemmatizer()
        .lemmatize(new String[] {"cats"}, new String[] {""}))
        .containsExactly("cat");
    assertThat(environment.loadWarnings())
        .anySatisfy(w -> assertThat(w)
            .contains("OPENNLP_MORFOLOGIK_DICT").contains("OPENNLP_LEMMATIZER_DICT"));
  }

  @Test
  void missingInfoSiblingDisablesTheBackendWithAWarning(@TempDir Path tempDir)
      throws Exception {
    final Path dict = tempDir.resolve("english.dict");
    Files.copy(morfologikDict(), dict);

    final PipelineEnvironment environment =
        PipelineEnvironment.load(configWithMorfologik(dict));

    assertThat(environment.lemmatizer()).isNull();
    assertThat(environment.loadWarnings())
        .anySatisfy(w -> assertThat(w)
            .contains("OPENNLP_MORFOLOGIK_DICT").contains("english.info"));
  }

  @Test
  void missingDictFileFailsStartup(@TempDir Path tempDir) {
    assertThatThrownBy(() -> PipelineEnvironment.load(
        configWithMorfologik(tempDir.resolve("no-such.dict"))))
        .isInstanceOfSatisfying(IllegalStateException.class, e ->
            assertThat(e.getMessage()).contains("OPENNLP_MORFOLOGIK_DICT"));
  }

  @Test
  void capabilitiesReportTheMorfologikBackendAsLemmatizer() {
    assertThat(stub.getCapabilities(GetCapabilitiesRequest.getDefaultInstance())
        .getLemmatizerAvailable()).isTrue();
  }
}
