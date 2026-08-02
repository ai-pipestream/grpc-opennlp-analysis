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

import java.util.List;

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import ai.pipestream.opennlp.analysis.config.ServiceConfig;
import ai.pipestream.opennlp.analysis.pipeline.PipelineEnvironment;
import ai.pipestream.opennlp.analysis.v1.AnalysisOptions;
import ai.pipestream.opennlp.analysis.v1.AnalysisServiceGrpc;
import ai.pipestream.opennlp.analysis.v1.AnalyzeRequest;
import ai.pipestream.opennlp.analysis.v1.AnalyzeResponse;
import ai.pipestream.opennlp.analysis.v1.CorefMention;
import ai.pipestream.opennlp.analysis.v1.DependencyArc;
import ai.pipestream.opennlp.analysis.v1.Entity;
import ai.pipestream.opennlp.analysis.v1.RelationOptions;
import ai.pipestream.opennlp.analysis.v1.Token;
import io.grpc.ManagedChannel;
import io.grpc.Server;
import io.grpc.inprocess.InProcessChannelBuilder;
import io.grpc.inprocess.InProcessServerBuilder;
import opennlp.tools.depparse.FeedforwardDependencyModel;
import opennlp.tools.namefind.TokenNameFinderModel;
import opennlp.tools.postag.POSModel;

/**
 * The model-backed happy paths, end to end over in-process gRPC: POS tagging,
 * coreference, dependency parsing, relation extraction, and geocoding against
 * tiny in-memory trained models ({@link TestModels}) that memorize the exact
 * sentences asserted on. {@link ModelDependentLayersTest} covers the same
 * layers when the models are missing.
 *
 * <p>Two servers are needed because a name finder carries a single entity
 * type: the main server tags persons (coref, relations), the geo server tags
 * locations (geocoding against the bundled gazetteer).</p>
 */
class ModelBackedLayersTest {

  private static Server server;
  private static ManagedChannel channel;
  private static AnalysisServiceGrpc.AnalysisServiceBlockingStub stub;

  private static Server geoServer;
  private static ManagedChannel geoChannel;
  private static AnalysisServiceGrpc.AnalysisServiceBlockingStub geoStub;

  private static POSModel posModel;

  @BeforeAll
  static void setUp() throws Exception {
    posModel = TestModels.trainPosModel();
    final TokenNameFinderModel personNer = TestModels.trainNerModel("person");
    final TokenNameFinderModel locationNer = TestModels.trainNerModel("location");
    final FeedforwardDependencyModel dependencyModel = TestModels.trainDependencyModel();

    final PipelineEnvironment environment = new PipelineEnvironment(
        null, null, posModel, personNer, null, null, null, null, dependencyModel,
        List.of());
    final PipelineEnvironment geoEnvironment = new PipelineEnvironment(
        null, null, posModel, locationNer, null, null, null, null, null, List.of());

    final String name = InProcessServerBuilder.generateName();
    server = InProcessServerBuilder.forName(name).directExecutor()
        .addService(new AnalysisServiceImpl(environment,
            new ServiceConfig(0, 1024 * 1024, null, null, null, null, null, null)))
        .build().start();
    channel = InProcessChannelBuilder.forName(name).directExecutor().build();
    stub = AnalysisServiceGrpc.newBlockingStub(channel);

    final String geoName = InProcessServerBuilder.generateName();
    geoServer = InProcessServerBuilder.forName(geoName).directExecutor()
        .addService(new AnalysisServiceImpl(geoEnvironment,
            new ServiceConfig(0, 1024 * 1024, null, null, null, null, null, null)))
        .build().start();
    geoChannel = InProcessChannelBuilder.forName(geoName).directExecutor().build();
    geoStub = AnalysisServiceGrpc.newBlockingStub(geoChannel);
  }

  @AfterAll
  static void tearDown() {
    channel.shutdownNow();
    server.shutdownNow();
    geoChannel.shutdownNow();
    geoServer.shutdownNow();
  }

  private static AnalyzeResponse analyze(
      AnalysisServiceGrpc.AnalysisServiceBlockingStub service, String text,
      AnalysisOptions.Builder options) {
    return service.analyze(AnalyzeRequest.newBuilder()
        .setText(text)
        .setOptions(options)
        .build());
  }

  @Test
  void posTagsComeBackAsUdTags() {
    final AnalyzeResponse response = analyze(stub, TestModels.COREF_SENTENCE_1,
        AnalysisOptions.newBuilder()
            .setSentenceDetection(true)
            .setPosTags(true));

    assertThat(response.getWarningsList()).isEmpty();
    // The model was trained on Penn tags; the service's default UD tag format
    // maps them (NNP -> PROPN, VBD -> VERB, DT -> DET, NN -> NOUN).
    assertThat(response.getTokensList()).extracting(Token::getText)
        .containsExactly("John", "Smith", "won", "the", "case");
    assertThat(response.getTokensList()).extracting(Token::getPos)
        .containsExactly("PROPN", "PROPN", "VERB", "DET", "NOUN");
  }

  @Test
  void posTagsImplySentenceDetectionWithAWarning() {
    // The POS annotator requires the sentences layer; the pipeline enables
    // sentence detection itself (the embeddings precedent) instead of failing
    // pipeline assembly.
    final AnalyzeResponse response = analyze(stub, TestModels.COREF_TEXT,
        AnalysisOptions.newBuilder().setPosTags(true));

    assertThat(response.getSentencesCount()).isEqualTo(2);
    assertThat(response.getTokensList()).allSatisfy(t -> assertThat(t.getPos()).isNotEmpty());
    assertThat(response.getWarningsList())
        .singleElement()
        .satisfies(w -> assertThat(w).contains("sentence_detection was enabled"));
  }

  @Test
  void corefLinksPronounToPersonEntityAcrossSentences() {
    final AnalyzeResponse response = analyze(stub, TestModels.COREF_TEXT,
        AnalysisOptions.newBuilder()
            .setSentenceDetection(true)
            .setCoref(true));

    assertThat(response.getWarningsList()).isEmpty();
    assertThat(response.getEntitiesList()).extracting(Entity::getText)
        .containsExactly("John Smith");
    assertThat(response.getEntitiesList()).extracting(Entity::getType)
        .containsExactly("person");

    final List<CorefMention> mentions = response.getCorefMentionsList();
    final CorefMention entityMention = mentions.stream()
        .filter(m -> m.getKind().equals("entity"))
        .findFirst().orElseThrow();
    assertThat(TestModels.COREF_TEXT.substring(
        entityMention.getSpan().getStart(), entityMention.getSpan().getEnd()))
        .isEqualTo("John Smith");
    assertThat(entityMention.getEntity()).isEqualTo(0);

    final CorefMention pronoun = mentions.stream()
        .filter(m -> m.getKind().equals("pronoun"))
        .findFirst().orElseThrow();
    assertThat(TestModels.COREF_TEXT.substring(
        pronoun.getSpan().getStart(), pronoun.getSpan().getEnd()))
        .isEqualTo("He");
    assertThat(pronoun.getEntity()).isEqualTo(-1);
    assertThat(pronoun.getChain()).isEqualTo(entityMention.getChain());
  }

  @Test
  void dependencyParseReproducesTheMemorizedTree() {
    final AnalyzeResponse response = analyze(stub, TestModels.RELATION_SENTENCE,
        AnalysisOptions.newBuilder()
            .setSentenceDetection(true)
            .setDependencyParse(true));

    assertThat(response.getWarningsList()).isEmpty();
    assertThat(response.getDependenciesCount())
        .isEqualTo(response.getTokensCount())
        .isEqualTo(TestModels.RELATION_TOKENS.length);

    final List<DependencyArc> arcs = response.getDependenciesList();
    // One arc per token, in token order, exactly one root.
    assertThat(arcs).extracting(DependencyArc::getDependent)
        .containsExactly(0, 1, 2, 3, 4);
    assertThat(arcs.stream().filter(a -> a.getHead() == -1)).hasSize(1);
    // The feedforward model memorized this exact sentence in training.
    assertThat(arcs).extracting(DependencyArc::getHead)
        .containsExactly(TestModels.RELATION_HEADS[0], TestModels.RELATION_HEADS[1],
            TestModels.RELATION_HEADS[2], TestModels.RELATION_HEADS[3],
            TestModels.RELATION_HEADS[4]);
    assertThat(arcs).extracting(DependencyArc::getRelation)
        .containsExactly(TestModels.RELATION_RELS);
    for (int i = 0; i < arcs.size(); i++) {
      final DependencyArc arc = arcs.get(i);
      assertThat(TestModels.RELATION_SENTENCE.substring(
          arc.getSpan().getStart(), arc.getSpan().getEnd()))
          .isEqualTo(TestModels.RELATION_TOKENS[i]);
    }
  }

  @Test
  void relationsMatchDependencyPathBetweenEntities() {
    final AnalyzeResponse response = analyze(stub, TestModels.RELATION_SENTENCE,
        AnalysisOptions.newBuilder()
            .setSentenceDetection(true)
            .setRelations(RelationOptions.newBuilder()
                .addPatterns(RelationOptions.Pattern.newBuilder()
                    .setType("founded")
                    .setPath("<nsubj >obj")
                    .setTrigger("founded"))));

    assertThat(response.getWarningsList()).isEmpty();
    // relations implies dependency_parse and ner, so both layers surface too.
    assertThat(response.getDependenciesCount()).isEqualTo(response.getTokensCount());
    assertThat(response.getEntitiesList()).extracting(Entity::getText)
        .containsExactly("John Smith", "Acme Corp");

    assertThat(response.getRelationsList()).hasSize(1);
    final var relation = response.getRelations(0);
    assertThat(relation.getType()).isEqualTo("founded");
    assertThat(response.getEntities(relation.getSubject()).getText()).isEqualTo("John Smith");
    assertThat(response.getEntities(relation.getObject()).getText()).isEqualTo("Acme Corp");
    assertThat(TestModels.RELATION_SENTENCE.substring(
        relation.getSpan().getStart(), relation.getSpan().getEnd()))
        .isEqualTo(TestModels.RELATION_SENTENCE);
  }

  @Test
  void capabilitiesReportTheConfiguredModels() {
    final var capabilities =
        stub.getCapabilities(ai.pipestream.opennlp.analysis.v1.GetCapabilitiesRequest
            .getDefaultInstance());

    assertThat(capabilities.getPosTagsAvailable()).isTrue();
    assertThat(capabilities.getNerAvailable()).isTrue();
    assertThat(capabilities.getDependencyParseAvailable()).isTrue();
    // The OpenNLP library version comes from the opennlp-api jar metadata.
    assertThat(capabilities.getOpennlpVersion()).isEqualTo("0.0.0-preview-SNAPSHOT");
    // No lattice/SentencePiece files are configured on this server.
    assertThat(capabilities.getLatticeAvailable()).isFalse();
    assertThat(capabilities.getSentencepieceAvailable()).isFalse();
  }

  @Test
  void posModelRoundTripsThroughAModelFile(@org.junit.jupiter.api.io.TempDir
                                           java.nio.file.Path tempDir)
      throws Exception {
    // The production load path (OPENNLP_POS_MODEL): serialize, then reload
    // from disk through BaseModel. This is the path the fork's broken
    // opennlp.version resource used to break; the build's generated version
    // resource makes it work.
    final java.nio.file.Path modelFile = tempDir.resolve("pos.bin");
    try (var out = java.nio.file.Files.newOutputStream(modelFile)) {
      posModel.serialize(out);
    }

    final POSModel reloaded = new POSModel(modelFile);
    final String[] tags = new opennlp.tools.postag.POSTaggerME(reloaded)
        .tag(new String[] {"John", "Smith", "won", "the", "case"});
    assertThat(tags).containsExactly("PROPN", "PROPN", "VERB", "DET", "NOUN");
  }

  @Test
  void geoResolvesLocationEntityAgainstBundledGazetteer() {
    final AnalyzeResponse response = analyze(geoStub, TestModels.GEO_SENTENCE,
        AnalysisOptions.newBuilder()
            .setSentenceDetection(true)
            .setGeo(true));

    assertThat(response.getWarningsList()).isEmpty();
    // geo implies ner, so the location entity surfaces as well.
    assertThat(response.getEntitiesList()).extracting(Entity::getText)
        .containsExactly("Paris");
    assertThat(response.getEntitiesList()).extracting(Entity::getType)
        .containsExactly("location");

    assertThat(response.getLocationsList()).isNotEmpty();
    assertThat(response.getLocationsList()).anySatisfy(location -> {
      assertThat(location.getName()).isEqualTo("Paris");
      assertThat(location.getCountryCode()).isNotBlank();
      assertThat(location.getConfidence()).isBetween(0.0, 1.0);
      assertThat(TestModels.GEO_SENTENCE.substring(
          location.getSpan().getStart(), location.getSpan().getEnd()))
          .isEqualTo("Paris");
    });

    assertThat(response.getRegionsList()).isNotEmpty();
    assertThat(response.getRegionsList()).allSatisfy(vote -> {
      assertThat(vote.getCountryCode()).isNotBlank();
      assertThat(vote.getShare()).isPositive();
    });
  }
}
