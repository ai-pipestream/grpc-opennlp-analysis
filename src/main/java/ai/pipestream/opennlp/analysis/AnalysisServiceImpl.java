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

import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ai.pipestream.opennlp.analysis.config.ServiceConfig;
import ai.pipestream.opennlp.analysis.pipeline.AnalysisPipeline;
import ai.pipestream.opennlp.analysis.pipeline.PipelineEnvironment;
import ai.pipestream.opennlp.analysis.pipeline.PipelineOptions;
import ai.pipestream.opennlp.analysis.v1.AnalysisOptions;
import ai.pipestream.opennlp.analysis.v1.AnalysisServiceGrpc;
import ai.pipestream.opennlp.analysis.v1.AnalyzeRequest;
import ai.pipestream.opennlp.analysis.v1.AnalyzeResponse;
import ai.pipestream.opennlp.analysis.v1.ChunkEmbedding;
import ai.pipestream.opennlp.analysis.v1.Entity;
import ai.pipestream.opennlp.analysis.v1.GetCapabilitiesRequest;
import ai.pipestream.opennlp.analysis.v1.GetCapabilitiesResponse;
import ai.pipestream.opennlp.analysis.v1.Span;
import ai.pipestream.opennlp.analysis.v1.TermVector;
import ai.pipestream.opennlp.analysis.v1.TermVectorOptions;
import ai.pipestream.opennlp.analysis.v1.Token;
import io.grpc.Status;
import io.grpc.stub.StreamObserver;
import opennlp.tools.document.Annotation;
import opennlp.tools.document.Document;
import opennlp.tools.document.Layers;
import opennlp.tools.stemmer.StemmerAnnotator;
import opennlp.tools.termvector.TermVectorAnnotator;

/**
 * gRPC front end of the analysis service. Validates requests, resolves (and
 * caches) one shared {@link AnalysisPipeline} per distinct option-set, and maps
 * the annotated {@link Document} onto the proto response.
 *
 * <p>Errors on the wire are clean gRPC statuses — INVALID_ARGUMENT for bad
 * input, INTERNAL for analysis failures — never stack traces.</p>
 */
public final class AnalysisServiceImpl extends AnalysisServiceGrpc.AnalysisServiceImplBase {

  private static final Logger LOG = LoggerFactory.getLogger(AnalysisServiceImpl.class);

  private final PipelineEnvironment environment;
  private final ServiceConfig config;
  private final ConcurrentHashMap<PipelineOptions, AnalysisPipeline> pipelines =
      new ConcurrentHashMap<>();

  /**
   * @param environment shared model resources, must not be {@code null}
   * @param config server configuration, must not be {@code null}
   */
  public AnalysisServiceImpl(PipelineEnvironment environment, ServiceConfig config) {
    if (environment == null || config == null) {
      throw new IllegalArgumentException("environment and config must not be null");
    }
    this.environment = environment;
    this.config = config;
  }

  @Override
  public void analyze(AnalyzeRequest request, StreamObserver<AnalyzeResponse> observer) {
    try {
      final String text = validatedText(request);
      final PipelineOptions options = OptionsMapper.fromProto(request.getOptions());
      final AnalysisPipeline pipeline =
          pipelines.computeIfAbsent(options, o -> AnalysisPipeline.create(o, environment));
      final Document document = pipeline.analyze(text);
      observer.onNext(toResponse(document, pipeline));
      observer.onCompleted();
    } catch (io.grpc.StatusRuntimeException e) {
      observer.onError(e);
    } catch (IllegalArgumentException e) {
      // Pipeline/layer contract violations caused by the request itself.
      observer.onError(Status.INVALID_ARGUMENT
          .withDescription(e.getMessage()).asRuntimeException());
    } catch (RuntimeException e) {
      LOG.error("Analysis failed", e);
      observer.onError(Status.INTERNAL
          .withDescription("analysis failed: " + e.getMessage()).asRuntimeException());
    }
  }

  @Override
  public void getCapabilities(GetCapabilitiesRequest request,
                              StreamObserver<GetCapabilitiesResponse> observer) {
    observer.onNext(GetCapabilitiesResponse.newBuilder()
        .setOpennlpVersion(Versions.opennlp())
        .setServiceVersion(Versions.service())
        .setEmbeddingsEnabled(environment.embeddingModel() != null)
        .setEmbeddingsModelDir(
            environment.embeddingsDirDescription() != null
                ? environment.embeddingsDirDescription() : "")
        .setPosTagsAvailable(environment.posModel() != null)
        .setNerAvailable(environment.nerModel() != null)
        .setMaxTextBytes(config.maxTextBytes())
        .addAllStemmers(java.util.Arrays.stream(AnalysisOptions.Stemmer.values())
            .filter(s -> s != AnalysisOptions.Stemmer.STEMMER_UNSPECIFIED
                && s != AnalysisOptions.Stemmer.UNRECOGNIZED)
            .map(Enum::name).toList())
        .addAllNormalizerRungs(java.util.Arrays.stream(
                TermVectorOptions.NormalizerRung.values())
            .filter(r -> r != TermVectorOptions.NormalizerRung
                .NORMALIZER_RUNG_UNSPECIFIED
                && r != TermVectorOptions.NormalizerRung.UNRECOGNIZED)
            .map(Enum::name).toList())
        .addAllTokenizers(java.util.Arrays.stream(AnalysisOptions.Tokenizer.values())
            .filter(t -> t != AnalysisOptions.Tokenizer.TOKENIZER_UNSPECIFIED
                && t != AnalysisOptions.Tokenizer.UNRECOGNIZED)
            .map(Enum::name).toList())
        .addAllWarnings(environment.loadWarnings())
        .build());
    observer.onCompleted();
  }

  /** Visible for testing: how many distinct pipelines are currently cached. */
  int cachedPipelineCount() {
    return pipelines.size();
  }

  private String validatedText(AnalyzeRequest request) {
    final String text = request.getText();
    if (text.isEmpty()) {
      throw Status.INVALID_ARGUMENT
          .withDescription("text must not be empty").asRuntimeException();
    }
    final int bytes = text.getBytes(StandardCharsets.UTF_8).length;
    if (bytes > config.maxTextBytes()) {
      throw Status.INVALID_ARGUMENT
          .withDescription("text is " + bytes + " bytes, above the limit of "
              + config.maxTextBytes() + " bytes").asRuntimeException();
    }
    return text;
  }

  /**
   * Maps the annotated document onto the response. Every span is copied
   * through unchanged: the layers are already in original text coordinates —
   * that is the offset-fidelity guarantee of the contract.
   */
  private static AnalyzeResponse toResponse(Document document, AnalysisPipeline pipeline) {
    final AnalyzeResponse.Builder response = AnalyzeResponse.newBuilder();
    final CharSequence text = document.text();

    for (Annotation<String> sentence : document.get(Layers.SENTENCES)) {
      response.addSentences(span(sentence.span()));
    }

    final Map<opennlp.tools.util.Span, String> posBySpan = bySpan(
        document.get(Layers.POS_TAGS));
    final Map<opennlp.tools.util.Span, String> stemBySpan = bySpan(
        document.get(StemmerAnnotator.STEMS));

    final List<Annotation<String>> tokens = document.get(Layers.TOKENS);
    for (Annotation<String> token : tokens) {
      final Token.Builder out = Token.newBuilder()
          .setSpan(span(token.span()))
          .setText(token.span().getCoveredText(text).toString());
      final String pos = posBySpan.get(token.span());
      if (pos != null) {
        out.setPos(pos);
      }
      response.addTokens(out);
      final String stem = stemBySpan.get(token.span());
      if (stem != null) {
        response.addStems(stem);
      }
    }

    for (Annotation<String> entity : document.get(Layers.ENTITIES)) {
      response.addEntities(Entity.newBuilder()
          .setSpan(span(entity.span()))
          .setType(entity.value())
          .setText(entity.span().getCoveredText(text).toString()));
    }

    for (Annotation<opennlp.tools.termvector.TermVector> annotation
        : document.get(TermVectorAnnotator.TERM_VECTORS)) {
      final opennlp.tools.termvector.TermVector vector = annotation.value();
      final TermVector.Builder out = TermVector.newBuilder()
          .setTerm(vector.term())
          .setFrequency(vector.frequency());
      if (vector.spans() != null) {
        vector.spans().forEach(occurrence -> out.addOccurrences(span(occurrence)));
      }
      response.addTermVectors(out);
    }

    if (pipeline.embeddingLayer() != null) {
      for (Annotation<float[]> embedding : document.get(pipeline.embeddingLayer())) {
        final ChunkEmbedding.Builder out = ChunkEmbedding.newBuilder()
            .setSpan(span(embedding.span()))
            .setText(embedding.span().getCoveredText(text).toString());
        for (float component : embedding.value()) {
          out.addVector(component);
        }
        response.addEmbeddings(out);
      }
    }

    response.addAllWarnings(pipeline.warnings());
    return response.build();
  }

  private static Span span(opennlp.tools.util.Span span) {
    return Span.newBuilder().setStart(span.getStart()).setEnd(span.getEnd()).build();
  }

  private static Map<opennlp.tools.util.Span, String> bySpan(
      List<Annotation<String>> annotations) {
    final Map<opennlp.tools.util.Span, String> map = new HashMap<>();
    for (Annotation<String> annotation : annotations) {
      map.put(annotation.span(), annotation.value());
    }
    return map;
  }
}
