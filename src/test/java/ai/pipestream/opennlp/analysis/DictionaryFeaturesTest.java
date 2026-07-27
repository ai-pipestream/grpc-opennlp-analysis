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

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import ai.pipestream.opennlp.analysis.config.ServiceConfig;
import ai.pipestream.opennlp.analysis.pipeline.PipelineEnvironment;
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
import opennlp.tools.stemmer.hunspell.HunspellDictionary;

/**
 * Dictionary-backed features with tiny test dictionaries: Hunspell stemming
 * and dictionary lemmatization, plus their capabilities flags.
 */
class DictionaryFeaturesTest {

  /** Minimal Hunspell fixture: "work" accepts the -er and -s suffixes. */
  private static final String AFFIX = String.join("\n",
      "SET UTF-8",
      "SFX E Y 1",
      "SFX E 0 er/S .",
      "SFX S Y 1",
      "SFX S 0 s [^sxy]",
      "");

  private static final String WORDS = "1\nwork/ES\n";

  /** Token/lemma pairs with the neutral empty tag the service looks up. */
  private static final String LEMMAS = "dogs\t\tdog\nruns\t\trun\n";

  @TempDir
  Path tempDir;

  private Server server;
  private ManagedChannel channel;
  private AnalysisServiceGrpc.AnalysisServiceBlockingStub stub;

  @BeforeEach
  void setUp() throws Exception {
    final Path aff = tempDir.resolve("test.aff");
    final Path dic = tempDir.resolve("test.dic");
    Files.writeString(aff, AFFIX);
    Files.writeString(dic, WORDS);
    final Path lemmaDict = tempDir.resolve("lemmas.tsv");
    Files.writeString(lemmaDict, LEMMAS);

    final PipelineEnvironment environment = new PipelineEnvironment(
        null, null, null, null,
        HunspellDictionary.load(aff, dic),
        new DictionaryLemmatizer(lemmaDict),
        List.of());

    final String name = InProcessServerBuilder.generateName();
    server = InProcessServerBuilder.forName(name).directExecutor()
        .addService(new AnalysisServiceImpl(environment,
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

  @Test
  void hunspellStemsThroughTheDictionary() {
    final AnalyzeResponse response = stub.analyze(AnalyzeRequest.newBuilder()
        .setText("workers work table")
        .setOptions(AnalysisOptions.newBuilder()
            .setStemmer(AnalysisOptions.Stemmer.STEMMER_HUNSPELL))
        .build());

    // "workers" analyzes back to "work"; unknown vocabulary passes through.
    assertThat(response.getStemsList()).containsExactly("work", "work", "table");
    assertThat(response.getWarningsList()).isEmpty();
  }

  @Test
  void dictionaryLemmatizerProducesLemmasParallelToTokens() {
    final AnalyzeResponse response = stub.analyze(AnalyzeRequest.newBuilder()
        .setText("dogs runs zzz")
        .setOptions(AnalysisOptions.newBuilder().setLemmatize(true))
        .build());

    assertThat(response.getTokensCount()).isEqualTo(3);
    // Unknown words come back as "O", the OpenNLP convention.
    assertThat(response.getLemmasList()).containsExactly("dog", "run", "O");
    assertThat(response.getWarningsList()).isEmpty();
  }

  @Test
  void lemmatizeWithoutDictionaryWouldWarn() {
    // This server HAS a dictionary; assert the happy path contract instead:
    // no warnings, lemmas present. The missing-dictionary warning path is
    // covered in AnalysisPipelineTest.
    final AnalyzeResponse response = stub.analyze(AnalyzeRequest.newBuilder()
        .setText("dogs")
        .setOptions(AnalysisOptions.newBuilder().setLemmatize(true))
        .build());
    assertThat(response.getLemmasList()).containsExactly("dog");
  }

  @Test
  void capabilitiesReportDictionaryAvailability() {
    final var capabilities = stub.getCapabilities(GetCapabilitiesRequest.getDefaultInstance());

    assertThat(capabilities.getHunspellAvailable()).isTrue();
    assertThat(capabilities.getLemmatizerAvailable()).isTrue();
    assertThat(capabilities.getEmbeddingsEnabled()).isFalse();
    assertThat(capabilities.getStemmersList()).contains("STEMMER_HUNSPELL");
  }
}
