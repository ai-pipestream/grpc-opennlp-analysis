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
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

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
import ai.pipestream.opennlp.analysis.v1.AnalyzeStreamDoc;
import ai.pipestream.opennlp.analysis.v1.AnalyzeStreamError;
import ai.pipestream.opennlp.analysis.v1.AnalyzeStreamRequest;
import ai.pipestream.opennlp.analysis.v1.AnalyzeStreamResponse;
import ai.pipestream.opennlp.analysis.v1.ChunkEmbedding;
import ai.pipestream.opennlp.analysis.v1.Entity;
import ai.pipestream.opennlp.analysis.v1.GetCapabilitiesRequest;
import ai.pipestream.opennlp.analysis.v1.GetCapabilitiesResponse;
import ai.pipestream.opennlp.analysis.v1.Span;
import ai.pipestream.opennlp.analysis.v1.TermVector;
import ai.pipestream.opennlp.analysis.v1.TermVectorOptions;
import ai.pipestream.opennlp.analysis.v1.Token;
import io.grpc.Status;
import io.grpc.stub.ServerCallStreamObserver;
import io.grpc.stub.StreamObserver;
import opennlp.tools.document.Annotation;
import opennlp.tools.document.Document;
import opennlp.tools.document.Layers;
import opennlp.tools.artifacts.ArtifactAnnotator;
import opennlp.tools.coref.CorefAnnotator;
import opennlp.tools.depparse.DependencyAnnotator;
import opennlp.tools.geo.DocumentRegionAnnotator;
import opennlp.tools.geo.GeocodeAnnotator;
import opennlp.tools.glossary.GlossaryAnnotator;
import opennlp.tools.lemmatizer.Lemmatizer;
import opennlp.tools.noise.NoiseAnnotator;
import opennlp.tools.pii.PiiAnnotator;
import opennlp.tools.relation.RelationAnnotator;
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

  // AnalyzeStream workers, shared across streams. The in-flight window each
  // stream grants through manual flow control is derived from this pool's
  // size, so capacity is declared exactly where the CPU lives instead of
  // being guessed client-side.
  private final ExecutorService streamWorkers;
  private final int streamWindow;

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
    final int workers = config.resolvedStreamWorkers();
    final AtomicInteger names = new AtomicInteger();
    this.streamWorkers = Executors.newFixedThreadPool(workers, runnable -> {
      final Thread thread = new Thread(runnable,
          "analyze-stream-" + names.incrementAndGet());
      // Daemon: the gRPC server's lifecycle is the process lifecycle; the
      // pool must never keep a shutting-down JVM alive.
      thread.setDaemon(true);
      return thread;
    });
    this.streamWindow = 2 * workers;
  }

  @Override
  public void analyze(AnalyzeRequest request, StreamObserver<AnalyzeResponse> observer) {
    try {
      final String text = validatedText(request.getText());
      final PipelineOptions options = OptionsMapper.fromProto(request.getOptions());
      final AnalysisPipeline pipeline =
          pipelines.computeIfAbsent(options, o -> AnalysisPipeline.create(o, environment));
      final Document document = pipeline.analyze(text);
      observer.onNext(toResponse(document, pipeline, environment.lemmatizer()));
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
  public StreamObserver<AnalyzeStreamRequest> analyzeStream(
      StreamObserver<AnalyzeStreamResponse> responseObserver) {
    final ServerCallStreamObserver<AnalyzeStreamResponse> call =
        (ServerCallStreamObserver<AnalyzeStreamResponse>) responseObserver;
    // Manual flow control: the server grants inbound messages from its own
    // worker capacity. One grant for the options message; the window opens
    // once options resolve, and every completed document grants one more.
    call.disableAutoRequest();
    final StreamSession session = new StreamSession(call);
    call.setOnCancelHandler(session::cancel);
    call.request(1);
    return session;
  }

  /**
   * One AnalyzeStream call. Documents are analyzed on the shared worker pool
   * and answered in completion order; {@code sendLock} serializes writes
   * because {@link StreamObserver} is not thread-safe.
   */
  private final class StreamSession implements StreamObserver<AnalyzeStreamRequest> {

    private final ServerCallStreamObserver<AnalyzeStreamResponse> call;
    private final Object sendLock = new Object();
    private final AtomicInteger inFlight = new AtomicInteger();
    private final AtomicBoolean finished = new AtomicBoolean();
    private volatile AnalysisPipeline pipeline;
    private volatile boolean clientDone;
    private volatile boolean cancelled;

    private StreamSession(ServerCallStreamObserver<AnalyzeStreamResponse> call) {
      this.call = call;
    }

    @Override
    public void onNext(AnalyzeStreamRequest request) {
      if (cancelled) {
        return;
      }
      switch (request.getMsgCase()) {
        case OPTIONS -> onOptions(request.getOptions());
        case DOC -> onDoc(request.getDoc());
        default -> failStream(Status.INVALID_ARGUMENT
            .withDescription("message carries neither options nor doc"));
      }
    }

    private void onOptions(AnalysisOptions options) {
      if (pipeline != null) {
        failStream(Status.INVALID_ARGUMENT
            .withDescription("options may only be the first message of the stream"));
        return;
      }
      try {
        final PipelineOptions mapped = OptionsMapper.fromProto(options);
        pipeline = pipelines.computeIfAbsent(mapped,
            o -> AnalysisPipeline.create(o, environment));
      } catch (io.grpc.StatusRuntimeException e) {
        failStream(e.getStatus());
        return;
      } catch (IllegalArgumentException e) {
        failStream(Status.INVALID_ARGUMENT.withDescription(e.getMessage()));
        return;
      } catch (RuntimeException e) {
        LOG.error("Stream pipeline construction failed", e);
        failStream(Status.INTERNAL
            .withDescription("pipeline construction failed: " + e.getMessage()));
        return;
      }
      call.request(streamWindow);
    }

    private void onDoc(AnalyzeStreamDoc doc) {
      if (pipeline == null) {
        failStream(Status.INVALID_ARGUMENT
            .withDescription("the first message of the stream must carry options"));
        return;
      }
      inFlight.incrementAndGet();
      streamWorkers.execute(() -> analyzeOne(doc));
    }

    private void analyzeOne(AnalyzeStreamDoc doc) {
      try {
        if (!cancelled) {
          send(resultFor(doc));
        }
      } finally {
        inFlight.decrementAndGet();
        if (!cancelled) {
          call.request(1);
        }
        maybeComplete();
      }
    }

    private AnalyzeStreamResponse resultFor(AnalyzeStreamDoc doc) {
      final AnalyzeStreamResponse.Builder response =
          AnalyzeStreamResponse.newBuilder().setSequence(doc.getSequence());
      try {
        final String text = validatedText(doc.getText());
        final Document document = pipeline.analyze(text);
        response.setOk(toResponse(document, pipeline, environment.lemmatizer()));
      } catch (io.grpc.StatusRuntimeException e) {
        response.setError(AnalyzeStreamError.newBuilder()
            .setCode(e.getStatus().getCode().value())
            .setMessage(String.valueOf(e.getStatus().getDescription())));
      } catch (IllegalArgumentException e) {
        response.setError(AnalyzeStreamError.newBuilder()
            .setCode(Status.Code.INVALID_ARGUMENT.value())
            .setMessage(String.valueOf(e.getMessage())));
      } catch (RuntimeException e) {
        LOG.error("Stream analysis failed for sequence {}", doc.getSequence(), e);
        response.setError(AnalyzeStreamError.newBuilder()
            .setCode(Status.Code.INTERNAL.value())
            .setMessage("analysis failed: " + e.getMessage()));
      }
      return response.build();
    }

    @Override
    public void onError(Throwable t) {
      // The client aborted; in-flight workers see the flag and stay quiet.
      cancelled = true;
    }

    @Override
    public void onCompleted() {
      clientDone = true;
      maybeComplete();
    }

    private void cancel() {
      cancelled = true;
    }

    private void maybeComplete() {
      if (clientDone && !cancelled && inFlight.get() == 0
          && finished.compareAndSet(false, true)) {
        synchronized (sendLock) {
          call.onCompleted();
        }
      }
    }

    private void send(AnalyzeStreamResponse response) {
      synchronized (sendLock) {
        if (!cancelled && !finished.get()) {
          call.onNext(response);
        }
      }
    }

    private void failStream(Status status) {
      cancelled = true;
      if (finished.compareAndSet(false, true)) {
        synchronized (sendLock) {
          call.onError(status.asRuntimeException());
        }
      }
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
        .setHunspellAvailable(environment.hunspellDictionary() != null)
        .setLemmatizerAvailable(environment.lemmatizer() != null)
        .setLatticeAvailable(environment.mecabDictionary() != null)
        .setSentencepieceAvailable(environment.sentencePieceTokenizer() != null)
        .setDependencyParseAvailable(environment.depparseModel() != null)
        .setDualTermIdentityAvailable(true)
        .setSpellcheckAvailable(environment.spellcheckModel() != null)
        .setMaxTextBytes(config.maxTextBytes())
        .addAllStemmers(java.util.Arrays.stream(AnalysisOptions.Stemmer.values())
            .filter(s -> s != AnalysisOptions.Stemmer.STEMMER_UNSPECIFIED
                && s != AnalysisOptions.Stemmer.UNRECOGNIZED)
            .map(Enum::name).toList())
        .addAllNormalizerSteps(java.util.Arrays.stream(
                TermVectorOptions.NormalizerStep.values())
            .filter(r -> r != TermVectorOptions.NormalizerStep
                .NORMALIZER_STEP_UNSPECIFIED
                && r != TermVectorOptions.NormalizerStep.UNRECOGNIZED)
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

  private String validatedText(String text) {
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
  private static AnalyzeResponse toResponse(Document document, AnalysisPipeline pipeline,
                                            Lemmatizer lemmatizer) {
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

    // Lemmatization, service-side: the OpenNLP LemmatizerAnnotator requires
    // the POS layer (a model dependency), so the configured backend is joined
    // onto the tokens here instead. POS tags feed the lookup when a POS model
    // produced them (UD tags are adapted for the WordNet backend); otherwise
    // every token looks up tagless (the dictionary backend's neutral tag, the
    // WordNet backend's all-POS fallback). Unknown words come back as "O",
    // the OpenNLP convention.
    if (pipeline.options().lemmatize() && lemmatizer != null) {
      final String[] words = new String[tokens.size()];
      final String[] tags = new String[tokens.size()];
      for (int i = 0; i < tokens.size(); i++) {
        words[i] = tokens.get(i).span().getCoveredText(text).toString();
        tags[i] = posBySpan.getOrDefault(tokens.get(i).span(), "");
      }
      for (String lemma : lemmatizer.lemmatize(words, tags)) {
        response.addLemmas(lemma);
      }
    }

    for (Annotation<String> entity : document.get(Layers.ENTITIES)) {
      response.addEntities(Entity.newBuilder()
          .setSpan(span(entity.span()))
          .setType(entity.value())
          .setText(entity.span().getCoveredText(text).toString()));
    }

    for (Annotation<opennlp.tools.noise.NoiseSpan> finding
        : document.get(NoiseAnnotator.NOISE)) {
      response.addNoise(ai.pipestream.opennlp.analysis.v1.NoiseSpan.newBuilder()
          .setSpan(span(finding.value().span()))
          .setSeverity(finding.value().severity())
          .setScore(finding.value().score()));
    }

    for (Annotation<opennlp.tools.artifacts.TextArtifact> artifact
        : document.get(ArtifactAnnotator.ARTIFACTS)) {
      response.addArtifacts(ai.pipestream.opennlp.analysis.v1.TextArtifact.newBuilder()
          .setSpan(span(artifact.value().span()))
          .setType(artifact.value().type()));
    }

    for (Annotation<opennlp.tools.glossary.GlossaryMatch> match
        : document.get(GlossaryAnnotator.GLOSSARY)) {
      response.addGlossaryMatches(
          ai.pipestream.opennlp.analysis.v1.GlossaryMatch.newBuilder()
              .setSpan(span(match.value().span()))
              .setId(match.value().id())
              .setTerm(match.value().term()));
    }

    for (Annotation<opennlp.tools.pii.PiiMention> mention
        : document.get(PiiAnnotator.PII)) {
      response.addPii(ai.pipestream.opennlp.analysis.v1.PiiMention.newBuilder()
          .setSpan(span(mention.value().span()))
          .setType(mention.value().type())
          .setNormalized(mention.value().normalized()));
    }

    for (Annotation<opennlp.tools.coref.CorefMention> mention
        : document.get(CorefAnnotator.CHAINS)) {
      response.addCorefMentions(
          ai.pipestream.opennlp.analysis.v1.CorefMention.newBuilder()
              .setSpan(span(mention.span()))
              .setChain(mention.value().chain())
              .setKind(mention.value().kind())
              .setEntity(mention.value().entity()));
    }

    for (Annotation<opennlp.tools.depparse.DependencyArc> arc
        : document.get(DependencyAnnotator.DEPENDENCIES)) {
      response.addDependencies(
          ai.pipestream.opennlp.analysis.v1.DependencyArc.newBuilder()
              .setSpan(span(arc.span()))
              .setHead(arc.value().head())
              .setDependent(arc.value().dependent())
              .setRelation(arc.value().relation()));
    }

    for (Annotation<opennlp.tools.relation.RelationMention> relation
        : document.get(RelationAnnotator.RELATIONS)) {
      response.addRelations(
          ai.pipestream.opennlp.analysis.v1.RelationMention.newBuilder()
              .setSpan(span(relation.span()))
              .setType(relation.value().type())
              .setSubject(relation.value().subject())
              .setObject(relation.value().object()));
    }

    for (Annotation<opennlp.tools.geo.GeoResolution> location
        : document.get(GeocodeAnnotator.LOCATIONS)) {
      final opennlp.tools.geo.GeoResolution resolution = location.value();
      final ai.pipestream.opennlp.analysis.v1.GeoLocation.Builder out =
          ai.pipestream.opennlp.analysis.v1.GeoLocation.newBuilder()
              .setSpan(span(resolution.mention()))
              .setConfidence(resolution.confidence());
      final opennlp.tools.geo.GazetteerEntry entry = resolution.entry();
      if (entry != null) {
        out.setName(entry.name());
        if (entry.countryCode() != null) {
          out.setCountryCode(entry.countryCode());
        }
        if (entry.location() != null) {
          out.setLatitude(entry.location().latitude());
          out.setLongitude(entry.location().longitude());
        }
      }
      response.addLocations(out);
    }

    for (Annotation<opennlp.tools.geo.RegionVote> vote
        : document.get(DocumentRegionAnnotator.REGIONS)) {
      response.addRegions(ai.pipestream.opennlp.analysis.v1.RegionVote.newBuilder()
          .setCountryCode(vote.value().countryCode())
          .setShare(vote.value().share()));
    }

    final PipelineOptions.TermVectorSpec termVectorSpec = pipeline.options().termVectors();
    if (termVectorSpec != null
        && (termVectorSpec.source() == PipelineOptions.TermVectorSource.STEMS
            || termVectorSpec.source()
                == PipelineOptions.TermVectorSource.NORMALIZED_STEMS)) {
      // The stem IS the term identity: group every token under its stem, in
      // first-occurrence order, carrying the token spans in original text
      // coordinates. The pipeline guarantees the stems layer exists here
      // (the mapper rejects either stem source without a stemmer). Under
      // NORMALIZED_STEMS the stemmer it consulted was the normalizing
      // decorator, so the grouping key is already the folded stem and this
      // block needs no other change.
      final boolean full = termVectorSpec.mode() == PipelineOptions.TermVectorMode.FULL;
      final Map<String, List<opennlp.tools.util.Span>> spansByStem = new LinkedHashMap<>();
      for (Annotation<String> token : tokens) {
        final String stem = stemBySpan.get(token.span());
        if (stem != null) {
          spansByStem.computeIfAbsent(stem, s -> new ArrayList<>()).add(token.span());
        }
      }
      spansByStem.forEach((stem, spans) -> {
        final TermVector.Builder out = TermVector.newBuilder()
            .setTerm(stem)
            .setFrequency(spans.size());
        if (full) {
          spans.forEach(occurrence -> out.addOccurrences(span(occurrence)));
        }
        response.addTermVectors(out);
      });
      if (termVectorSpec.dualCased() && pipeline.casedStemmer() != null) {
        // The cased half of a dual-identity request: the same grouping, keyed
        // by the stem of the case-preserving normalized form. Same tokens,
        // same spans, one pass.
        final Map<String, List<opennlp.tools.util.Span>> spansByCasedStem =
            new LinkedHashMap<>();
        for (Annotation<String> token : tokens) {
          final String casedStem = pipeline.casedStemmer()
              .stem(token.span().getCoveredText(text)).toString();
          spansByCasedStem.computeIfAbsent(casedStem, s -> new ArrayList<>())
              .add(token.span());
        }
        spansByCasedStem.forEach((stem, spans) -> {
          final TermVector.Builder out = TermVector.newBuilder()
              .setTerm(stem)
              .setFrequency(spans.size());
          if (full) {
            spans.forEach(occurrence -> out.addOccurrences(span(occurrence)));
          }
          response.addCasedTermVectors(out);
        });
      }
    } else {
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
      if (termVectorSpec != null && termVectorSpec.dualCased()
          && pipeline.casedNormalizer() != null) {
        // Token-sourced dual identity: group per token under the
        // case-preserving normalized form, the same semantics the per-token
        // term vector annotator applies for the folded list.
        final boolean full =
            termVectorSpec.mode() == PipelineOptions.TermVectorMode.FULL;
        final Map<String, List<opennlp.tools.util.Span>> spansByCasedTerm =
            new LinkedHashMap<>();
        for (Annotation<String> token : tokens) {
          final String casedTerm = pipeline.casedNormalizer()
              .normalize(token.span().getCoveredText(text)).toString();
          spansByCasedTerm.computeIfAbsent(casedTerm, s -> new ArrayList<>())
              .add(token.span());
        }
        spansByCasedTerm.forEach((term, spans) -> {
          final TermVector.Builder out = TermVector.newBuilder()
              .setTerm(term)
              .setFrequency(spans.size());
          if (full) {
            spans.forEach(occurrence -> out.addOccurrences(span(occurrence)));
          }
          response.addCasedTermVectors(out);
        });
      }
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
