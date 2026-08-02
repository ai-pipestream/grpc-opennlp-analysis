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
import ai.pipestream.opennlp.analysis.v1.GetCapabilitiesResponse;
import ai.pipestream.opennlp.analysis.v1.TermVectorOptions;
import io.grpc.ManagedChannel;
import io.grpc.Status;
import io.grpc.StatusRuntimeException;
import io.grpc.inprocess.InProcessChannelBuilder;
import io.grpc.inprocess.InProcessServerBuilder;
import io.grpc.Server;

/**
 * In-process round-trip tests: the full service against an
 * {@link InProcessServerBuilder} channel, with no models configured.
 */
class AnalysisServiceImplTest {

  private static final ServiceConfig CONFIG =
      new ServiceConfig(0, 1024 * 1024, null, null, null, null, null, null);

  private String serverName;
  private Server server;
  private ManagedChannel channel;
  private AnalysisServiceGrpc.AnalysisServiceBlockingStub stub;
  private AnalysisServiceImpl service;

  @BeforeEach
  void setUp() throws Exception {
    serverName = InProcessServerBuilder.generateName();
    service = new AnalysisServiceImpl(PipelineEnvironment.empty(), CONFIG);
    server = InProcessServerBuilder.forName(serverName)
        .directExecutor()
        .addService(service)
        .build()
        .start();
    channel = InProcessChannelBuilder.forName(serverName).directExecutor().build();
    stub = AnalysisServiceGrpc.newBlockingStub(channel);
  }

  @AfterEach
  void tearDown() {
    channel.shutdownNow();
    server.shutdownNow();
  }

  @Test
  void analyzeRoundTripWithTermVectorsStemsAndSentences() {
    final AnalyzeResponse response = stub.analyze(AnalyzeRequest.newBuilder()
        .setText("Groß  groß  GROSS")
        .setOptions(AnalysisOptions.newBuilder()
            .setSentenceDetection(true)
            .setStemmer(AnalysisOptions.Stemmer.STEMMER_PORTER)
            .setTermVectors(TermVectorOptions.newBuilder()
                .setEnabled(true)
                .setMode(TermVectorOptions.Mode.MODE_FULL)
                .addSteps(TermVectorOptions.NormalizerStep.NORMALIZER_STEP_WHITESPACE)
                .addSteps(TermVectorOptions.NormalizerStep.NORMALIZER_STEP_FULL_CASE_FOLD)))
        .build());

    assertThat(response.getSentencesCount()).isEqualTo(1);
    assertThat(response.getTokensList())
        .extracting(t -> t.getText())
        .containsExactly("Groß", "groß", "GROSS");
    assertThat(response.getTokensList())
        .extracting(t -> t.getSpan().getStart() + "-" + t.getSpan().getEnd())
        .containsExactly("0-4", "6-10", "12-17");
    // Stemmers do not lowercase; porter leaves all three surface forms as-is.
    assertThat(response.getStemsList()).containsExactly("Groß", "groß", "GROSS");

    assertThat(response.getTermVectorsCount()).isEqualTo(1);
    final ai.pipestream.opennlp.analysis.v1.TermVector vector = response.getTermVectors(0);
    assertThat(vector.getTerm()).isEqualTo("gross");
    assertThat(vector.getFrequency()).isEqualTo(3);
    assertThat(vector.getOccurrencesList())
        .extracting(s -> s.getStart() + "-" + s.getEnd())
        .containsExactly("0-4", "6-10", "12-17");
  }

  @Test
  void mixedStepChainWithANonOffsetAwareStepWorksOverTheWire() {
    // NFKC (full-width fold) cannot report an alignment, so this chain runs
    // through the per-token annotator; FULL_CASE_FOLD composes with it. Both
    // surface forms collapse to one term with original-text occurrence spans.
    final AnalyzeResponse response = stub.analyze(AnalyzeRequest.newBuilder()
        .setText("ＣＯＵＲＴ court")
        .setOptions(AnalysisOptions.newBuilder()
            .setTermVectors(TermVectorOptions.newBuilder()
                .setEnabled(true)
                .setMode(TermVectorOptions.Mode.MODE_FULL)
                .addSteps(TermVectorOptions.NormalizerStep.NORMALIZER_STEP_NFKC)
                .addSteps(
                    TermVectorOptions.NormalizerStep.NORMALIZER_STEP_FULL_CASE_FOLD)))
        .build());

    assertThat(response.getTermVectorsCount()).isEqualTo(1);
    final ai.pipestream.opennlp.analysis.v1.TermVector vector = response.getTermVectors(0);
    assertThat(vector.getTerm()).isEqualTo("court");
    assertThat(vector.getFrequency()).isEqualTo(2);
    assertThat(vector.getOccurrencesList())
        .extracting(s -> s.getStart() + "-" + s.getEnd())
        .containsExactly("0-5", "6-11");
  }

  @Test
  void scoringOnlyOmitsOccurrencesButKeepsFrequencies() {    final AnalyzeResponse response = stub.analyze(AnalyzeRequest.newBuilder()
        .setText("Groß  groß  GROSS")
        .setOptions(AnalysisOptions.newBuilder()
            .setTermVectors(TermVectorOptions.newBuilder()
                .setEnabled(true)
                .setMode(TermVectorOptions.Mode.MODE_SCORING_ONLY)
                .addSteps(TermVectorOptions.NormalizerStep.NORMALIZER_STEP_WHITESPACE)
                .addSteps(TermVectorOptions.NormalizerStep.NORMALIZER_STEP_FULL_CASE_FOLD)))
        .build());

    assertThat(response.getTermVectorsCount()).isEqualTo(1);
    assertThat(response.getTermVectors(0).getFrequency()).isEqualTo(3);
    assertThat(response.getTermVectors(0).getOccurrencesCount()).isZero();
  }

  @Test
  void emptyTextIsRejected() {
    assertThatThrownBy(() -> stub.analyze(
        AnalyzeRequest.newBuilder().setText("").build()))
        .isInstanceOfSatisfying(StatusRuntimeException.class,
            e -> assertThat(e.getStatus().getCode())
                .isEqualTo(Status.Code.INVALID_ARGUMENT));
  }

  @Test
  void oversizedTextIsRejected() {
    final AnalysisServiceImpl tiny =
        new AnalysisServiceImpl(PipelineEnvironment.empty(),
            new ServiceConfig(0, 16, null, null, null, null, null, null));
    final String name = InProcessServerBuilder.generateName();
    Server tinyServer = null;
    ManagedChannel tinyChannel = null;
    try {
      tinyServer = InProcessServerBuilder.forName(name).directExecutor()
          .addService(tiny).build().start();
      tinyChannel = InProcessChannelBuilder.forName(name).directExecutor().build();
      final AnalysisServiceGrpc.AnalysisServiceBlockingStub tinyStub =
          AnalysisServiceGrpc.newBlockingStub(tinyChannel);

      assertThatThrownBy(() -> tinyStub.analyze(
          AnalyzeRequest.newBuilder().setText("this text is way too long").build()))
          .isInstanceOfSatisfying(StatusRuntimeException.class, e -> {
            assertThat(e.getStatus().getCode()).isEqualTo(Status.Code.INVALID_ARGUMENT);
            assertThat(e.getStatus().getDescription()).contains("limit");
          });
    } catch (Exception e) {
      throw new AssertionError(e);
    } finally {
      if (tinyChannel != null) {
        tinyChannel.shutdownNow();
      }
      if (tinyServer != null) {
        tinyServer.shutdownNow();
      }
    }
  }

  @Test
  void unavailableFeaturesWarnInsteadOfFailing() {
    final AnalyzeResponse response = stub.analyze(AnalyzeRequest.newBuilder()
        .setText("hello world")
        .setOptions(AnalysisOptions.newBuilder()
            .setPosTags(true)
            .setNer(true)
            .setLemmatize(true)
            .setStemmer(AnalysisOptions.Stemmer.STEMMER_HUNSPELL)
            .setEmbeddings(ai.pipestream.opennlp.analysis.v1.EmbeddingOptions.newBuilder()
                .setSource(ai.pipestream.opennlp.analysis.v1.EmbeddingOptions.Source
                    .SOURCE_SENTENCES)))
        .build());

    assertThat(response.getTokensCount()).isEqualTo(2);
    assertThat(response.getEntitiesCount()).isZero();
    assertThat(response.getEmbeddingsCount()).isZero();
    assertThat(response.getLemmasCount()).isZero();
    assertThat(response.getStemsCount()).isZero();
    assertThat(response.getWarningsList())
        .anySatisfy(w -> assertThat(w).contains("POS"))
        .anySatisfy(w -> assertThat(w).contains("NER"))
        .anySatisfy(w -> assertThat(w).contains("Hunspell"))
        .anySatisfy(w -> assertThat(w).contains("lemmatizer"))
        .anySatisfy(w -> assertThat(w).contains("embedding"));
  }

  @Test
  void identicalOptionSetsShareOneCachedPipeline() {
    final AnalyzeRequest request = AnalyzeRequest.newBuilder()
        .setText("hello world")
        .setOptions(AnalysisOptions.newBuilder()
            .setStemmer(AnalysisOptions.Stemmer.STEMMER_PORTER))
        .build();
    stub.analyze(request);
    stub.analyze(request);
    stub.analyze(AnalyzeRequest.newBuilder().setText("other text")
        .setOptions(AnalysisOptions.newBuilder()
            .setStemmer(AnalysisOptions.Stemmer.STEMMER_SNOWBALL_ENGLISH))
        .build());

    assertThat(service.cachedPipelineCount()).isEqualTo(2);
  }

  @Test
  void capabilitiesReportEmbeddingsDisabledAndAvailableOptions() {
    final GetCapabilitiesResponse capabilities =
        stub.getCapabilities(GetCapabilitiesRequest.getDefaultInstance());

    assertThat(capabilities.getOpennlpVersion()).isEqualTo("0.0.0-preview-SNAPSHOT");
    assertThat(capabilities.getEmbeddingsEnabled()).isFalse();
    assertThat(capabilities.getEmbeddingsModelDir()).isEmpty();
    assertThat(capabilities.getPosTagsAvailable()).isFalse();
    assertThat(capabilities.getNerAvailable()).isFalse();
    assertThat(capabilities.getHunspellAvailable()).isFalse();
    assertThat(capabilities.getLemmatizerAvailable()).isFalse();
    assertThat(capabilities.getMaxTextBytes()).isEqualTo(1024 * 1024);
    assertThat(capabilities.getStemmersList())
        .contains("STEMMER_PORTER", "STEMMER_LIGHT_GERMAN",
            "STEMMER_SNOWBALL_TURKISH", "STEMMER_MINIMAL_SWEDISH",
            "STEMMER_LIGHT_NORWEGIAN_NYNORSK", "STEMMER_HUNSPELL")
        .doesNotContain("STEMMER_UNSPECIFIED");
    assertThat(capabilities.getNormalizerStepsList())
        .contains("NORMALIZER_STEP_FULL_CASE_FOLD", "NORMALIZER_STEP_GERMAN_UMLAUT",
            "NORMALIZER_STEP_ELLIPSIS", "NORMALIZER_STEP_BULLETS",
            "NORMALIZER_STEP_EMOJI_TO_EMOTICON", "NORMALIZER_STEP_EMOTICON_TO_EMOJI",
            "NORMALIZER_STEP_NFKC", "NORMALIZER_STEP_NFC",
            "NORMALIZER_STEP_CONFUSABLE_SKELETON", "NORMALIZER_STEP_ACCENT_FOLD",
            "NORMALIZER_STEP_CASE_FOLD",
            "NORMALIZER_STEP_LINE_BREAK_PRESERVING_WHITESPACE",
            "NORMALIZER_STEP_URL", "NORMALIZER_STEP_NUMBER",
            "NORMALIZER_STEP_SOCIAL_MEDIA", "NORMALIZER_STEP_SHRINK");
    assertThat(capabilities.getTokenizersList())
        .containsExactlyInAnyOrder("TOKENIZER_WHITESPACE", "TOKENIZER_SIMPLE",
            "TOKENIZER_UAX29", "TOKENIZER_LATTICE", "TOKENIZER_SENTENCEPIECE");
    assertThat(capabilities.getLatticeAvailable()).isFalse();
    assertThat(capabilities.getSentencepieceAvailable()).isFalse();
    assertThat(capabilities.getDependencyParseAvailable()).isFalse();
  }
}
