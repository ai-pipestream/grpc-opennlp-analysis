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

import java.io.InputStream;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

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
import io.grpc.inprocess.InProcessChannelBuilder;
import io.grpc.inprocess.InProcessServerBuilder;
import opennlp.spellcheck.dictionary.SymSpellModel;
import opennlp.spellcheck.dictionary.SymSpellModels;
import opennlp.spellcheck.symspell.SymSpellConfig;

/**
 * The model-gated spell-check step: SymSpell correction inside the term
 * identity chain, its casing re-application, the skipped-with-warning
 * behavior when no model is configured, and environment loading from a
 * serialized model file.
 */
class SpellcheckStepTest {

  /** A four-word dictionary: enough to correct "dargons" and nothing else. */
  private static SymSpellModel tinyModel() {
    return new SymSpellModel("en", SymSpellConfig.defaultConfig(),
        Map.of("court", 1000L, "dungeons", 800L, "and", 5000L, "dragons", 900L),
        Map.of());
  }

  private static AnalysisServiceGrpc.AnalysisServiceBlockingStub stub(
      PipelineEnvironment environment, Server[] serverBox, ManagedChannel[] channelBox)
      throws Exception {
    final String name = InProcessServerBuilder.generateName();
    final Server server = InProcessServerBuilder.forName(name).directExecutor()
        .addService(new AnalysisServiceImpl(environment,
            new ServiceConfig(0, 1024 * 1024, null, null, null, null, null, null)))
        .build().start();
    final ManagedChannel channel =
        InProcessChannelBuilder.forName(name).directExecutor().build();
    serverBox[0] = server;
    channelBox[0] = channel;
    return AnalysisServiceGrpc.newBlockingStub(channel);
  }

  private static AnalyzeResponse analyze(
      AnalysisServiceGrpc.AnalysisServiceBlockingStub stub, String text,
      TermVectorOptions.NormalizerStep... steps) {
    final TermVectorOptions.Builder termVectors = TermVectorOptions.newBuilder()
        .setEnabled(true)
        .setSource(TermVectorOptions.Source.SOURCE_TOKENS)
        .setMode(TermVectorOptions.Mode.MODE_FULL);
    for (final TermVectorOptions.NormalizerStep step : steps) {
      termVectors.addSteps(step);
    }
    return stub.analyze(AnalyzeRequest.newBuilder()
        .setText(text)
        .setOptions(AnalysisOptions.newBuilder().setTermVectors(termVectors))
        .build());
  }

  @Test
  void spellcheckCorrectsDamagedTokensBeforeFolding() throws Exception {
    final Server[] serverBox = new Server[1];
    final ManagedChannel[] channelBox = new ManagedChannel[1];
    final AnalysisServiceGrpc.AnalysisServiceBlockingStub stub = stub(
        new PipelineEnvironment(null, null, null, null, null, null, null, null, null,
            tinyModel(), List.of()),
        serverBox, channelBox);
    try {
      final AnalyzeResponse response = analyze(stub, "dungeons and dargons",
          TermVectorOptions.NormalizerStep.NORMALIZER_STEP_SPELLCHECK,
          TermVectorOptions.NormalizerStep.NORMALIZER_STEP_FULL_CASE_FOLD);

      // "dargons" corrects to "dragons" (Damerau distance 1) and then folds;
      // dictionary words and short tokens pass through untouched.
      assertThat(response.getTermVectorsList())
          .extracting(v -> v.getTerm() + ":" + v.getFrequency())
          .containsExactlyInAnyOrder("dungeons:1", "and:1", "dragons:1");
      // The corrected term keeps the damaged token's original span.
      assertThat(response.getTermVectorsList().stream()
          .filter(v -> v.getTerm().equals("dragons"))
          .flatMap(v -> v.getOccurrencesList().stream())
          .map(s -> s.getStart() + "-" + s.getEnd()))
          .containsExactly("13-20");
      assertThat(response.getWarningsList()).isEmpty();
    } finally {
      channelBox[0].shutdownNow();
      serverBox[0].shutdownNow();
    }
  }

  @Test
  void correctionReappliesTheTokensCasing() throws Exception {
    final Server[] serverBox = new Server[1];
    final ManagedChannel[] channelBox = new ManagedChannel[1];
    final AnalysisServiceGrpc.AnalysisServiceBlockingStub stub = stub(
        new PipelineEnvironment(null, null, null, null, null, null, null, null, null,
            tinyModel(), List.of()),
        serverBox, channelBox);
    try {
      final AnalyzeResponse response = analyze(stub, "Dargons DARGONS",
          TermVectorOptions.NormalizerStep.NORMALIZER_STEP_SPELLCHECK);

      assertThat(response.getTermVectorsList())
          .extracting(v -> v.getTerm() + ":" + v.getFrequency())
          .containsExactlyInAnyOrder("Dragons:1", "DRAGONS:1");
    } finally {
      channelBox[0].shutdownNow();
      serverBox[0].shutdownNow();
    }
  }

  @Test
  void missingModelSkipsTheStepWithAWarning() throws Exception {
    final Server[] serverBox = new Server[1];
    final ManagedChannel[] channelBox = new ManagedChannel[1];
    final AnalysisServiceGrpc.AnalysisServiceBlockingStub stub =
        stub(PipelineEnvironment.empty(), serverBox, channelBox);
    try {
      final AnalyzeResponse response = analyze(stub, "dargons",
          TermVectorOptions.NormalizerStep.NORMALIZER_STEP_SPELLCHECK);

      assertThat(response.getWarningsList())
          .anySatisfy(w -> assertThat(w).contains("OPENNLP_SPELLCHECK_MODEL"));
      assertThat(response.getTermVectorsList())
          .extracting(v -> v.getTerm())
          .containsExactly("dargons");
      assertThat(stub.getCapabilities(GetCapabilitiesRequest.getDefaultInstance())
          .getSpellcheckAvailable()).isFalse();
    } finally {
      channelBox[0].shutdownNow();
      serverBox[0].shutdownNow();
    }
  }

  @Test
  void environmentLoadsASerializedModel(@TempDir Path tempDir) throws Exception {
    final Path modelFile = tempDir.resolve("spellcheck.bin");
    try (OutputStream out = Files.newOutputStream(modelFile)) {
      SymSpellModels.serialize(tinyModel(), out);
    }
    final ServiceConfig config = new ServiceConfig(0, 1024 * 1024, null, null, null,
        null, null, null, null, null, null, null, null, modelFile, 0);

    final PipelineEnvironment environment = PipelineEnvironment.load(config);

    assertThat(environment.spellcheckModel()).isNotNull();
    assertThat(environment.spellcheckModel().unigrams()).containsKey("dragons");
    assertThat(environment.loadWarnings()).noneMatch(w -> w.contains("SPELLCHECK"));

    // Round-trip sanity: reading the file back yields the same dictionary.
    try (InputStream in = Files.newInputStream(modelFile)) {
      assertThat(SymSpellModels.deserialize(in).unigrams()).containsKey("court");
    }
  }
}
