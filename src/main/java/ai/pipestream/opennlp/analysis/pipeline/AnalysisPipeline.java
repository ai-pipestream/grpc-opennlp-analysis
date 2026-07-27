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

package ai.pipestream.opennlp.analysis.pipeline;

import java.util.ArrayList;
import java.util.List;

import opennlp.tools.document.Document;
import opennlp.tools.document.DocumentAnalyzer;
import opennlp.tools.document.DocumentAnnotator;
import opennlp.tools.document.LayerKey;
import opennlp.tools.document.Layers;
import opennlp.tools.document.NameFinderAnnotator;
import opennlp.tools.document.POSTaggerAnnotator;
import opennlp.tools.document.SentenceDetectorAnnotator;
import opennlp.tools.document.TokenizerAnnotator;
import opennlp.tools.embeddings.EmbeddingAnnotator;
import opennlp.tools.namefind.NameFinderME;
import opennlp.tools.postag.POSTaggerME;
import opennlp.tools.sentdetect.NewlineSentenceDetector;
import opennlp.tools.stemmer.PorterStemmerFactory;
import opennlp.tools.stemmer.SharingStemmer;
import opennlp.tools.stemmer.Stemmer;
import opennlp.tools.stemmer.StemmerAnnotator;
import opennlp.tools.stemmer.StemmerFactory;
import opennlp.tools.stemmer.light.EnglishMinimalStemmer;
import opennlp.tools.stemmer.light.FrenchLightStemmer;
import opennlp.tools.stemmer.light.GermanLightStemmer;
import opennlp.tools.stemmer.light.SpanishLightStemmer;
import opennlp.tools.stemmer.snowball.SnowballStemmer;
import opennlp.tools.termvector.TermVectorAnnotator;
import opennlp.tools.tokenize.SimpleTokenizer;
import opennlp.tools.tokenize.WhitespaceTokenizer;
import opennlp.tools.util.normalizer.OffsetAwareNormalizer;
import opennlp.tools.util.normalizer.TextNormalizer;

/**
 * One compiled analysis pipeline for a distinct {@link PipelineOptions} set.
 * Holds a shared {@link DocumentAnalyzer} plus the metadata needed to read its
 * output back out. Construction is gRPC-free and unit-testable.
 *
 * <p>Thread safety: {@link DocumentAnalyzer}, {@code Document}, the aligned
 * normalizer, and the embedding model are thread-safe; stemmers are wrapped in
 * a {@link SharingStemmer} so each thread gets its own delegate; the
 * model-based POS tagger and name finder carry per-call adaptive state and are
 * therefore serialized behind a lock. A single instance serves concurrent
 * {@link #analyze(CharSequence)} calls.</p>
 */
public final class AnalysisPipeline {

  private final PipelineOptions options;
  private final DocumentAnalyzer analyzer;
  private final EmbeddingAnnotator embeddingAnnotator;
  private final List<String> warnings;

  private AnalysisPipeline(PipelineOptions options, DocumentAnalyzer analyzer,
                           EmbeddingAnnotator embeddingAnnotator, List<String> warnings) {
    this.options = options;
    this.analyzer = analyzer;
    this.embeddingAnnotator = embeddingAnnotator;
    this.warnings = warnings;
  }

  /**
   * Builds a pipeline for the given options against the shared environment.
   * Requested features whose model the environment lacks are skipped and
   * reported in {@link #warnings()}; the pipeline still serves everything else.
   *
   * @param options the option-set, must not be {@code null}
   * @param environment the shared server resources, must not be {@code null}
   * @return the compiled pipeline, never {@code null}
   */
  public static AnalysisPipeline create(PipelineOptions options, PipelineEnvironment environment) {
    if (options == null || environment == null) {
      throw new IllegalArgumentException("options and environment must not be null");
    }
    final List<String> warnings = new ArrayList<>();
    final DocumentAnalyzer.Builder builder = DocumentAnalyzer.builder();

    boolean sentenceDetection = options.sentenceDetection();
    if (options.embeddingSource() == PipelineOptions.EmbeddingSource.SENTENCES
        && !sentenceDetection) {
      sentenceDetection = true;
      warnings.add("embedding source SENTENCES requires sentence detection; "
          + "sentence_detection was enabled for this request");
    }
    if (sentenceDetection) {
      // Newline-based and model-free: sentence offsets are always exact and the
      // default pipeline needs zero downloads.
      builder.add(new SentenceDetectorAnnotator(new NewlineSentenceDetector()));
    }

    builder.add(switch (options.tokenizer()) {
      case WHITESPACE -> new TokenizerAnnotator(WhitespaceTokenizer.INSTANCE);
      case SIMPLE -> new TokenizerAnnotator(SimpleTokenizer.INSTANCE);
    });

    if (options.stemmer() != PipelineOptions.Stemmer.NONE) {
      builder.add(new StemmerAnnotator(threadSafeStemmer(options.stemmer())));
    }

    if (options.posTags()) {
      if (environment.posModel() != null) {
        // Guarded: the tagger keeps per-call adaptive state in its searcher.
        builder.add(new SynchronizedAnnotator(
            new POSTaggerAnnotator(new POSTaggerME(environment.posModel()))));
      } else {
        warnings.add("pos_tags was requested but no POS model is configured "
            + "(OPENNLP_POS_MODEL); pos stays empty");
      }
    }

    if (options.ner()) {
      if (environment.nerModel() != null) {
        // Guarded: the name finder clears adaptive data per call.
        builder.add(new SynchronizedAnnotator(
            new NameFinderAnnotator(new NameFinderME(environment.nerModel()))));
      } else {
        warnings.add("ner was requested but no NER model is configured "
            + "(OPENNLP_NER_MODEL); entities stays empty");
      }
    }

    if (options.termVectors() != null) {
      final PipelineOptions.TermVectorSpec spec = options.termVectors();
      builder.add(new TermVectorAnnotator(alignedNormalizer(spec.rungs()),
          spec.mode() == PipelineOptions.TermVectorMode.SCORING_ONLY
              ? TermVectorAnnotator.Mode.SCORING_ONLY : TermVectorAnnotator.Mode.FULL));
    }

    EmbeddingAnnotator embeddingAnnotator = null;
    if (options.embeddingSource() != null) {
      if (environment.embeddingModel() != null) {
        final LayerKey<String> source =
            options.embeddingSource() == PipelineOptions.EmbeddingSource.SENTENCES
                ? Layers.SENTENCES : Layers.TOKENS;
        embeddingAnnotator = new EmbeddingAnnotator(environment.embeddingModel(), source);
        builder.add(embeddingAnnotator);
      } else {
        warnings.add("embeddings were requested but no embedding model is loaded "
            + "(OPENNLP_EMBEDDINGS_DIR); embeddings stays empty");
      }
    }

    return new AnalysisPipeline(options, builder.build(), embeddingAnnotator,
        List.copyOf(warnings));
  }

  /**
   * Runs the pipeline over the given text.
   *
   * @param text the text to analyze, must not be {@code null}
   * @return the annotated document, never {@code null}
   */
  public Document analyze(CharSequence text) {
    return analyzer.analyze(text);
  }

  /**
   * @return the options this pipeline was built for, never {@code null}
   */
  public PipelineOptions options() {
    return options;
  }

  /**
   * @return the layer the embedding annotator provides, or {@code null} when
   *         this pipeline does not embed
   */
  public LayerKey<float[]> embeddingLayer() {
    return embeddingAnnotator != null ? embeddingAnnotator.layer() : null;
  }

  /**
   * @return requested-but-unavailable features, surfaced on every response
   *         produced through this pipeline; never {@code null}
   */
  public List<String> warnings() {
    return warnings;
  }

  /**
   * Builds the aligned normalizer for the requested rungs. Rungs apply in the
   * fixed order of the OpenNLP builder, independent of request order, so equal
   * rung sets always produce equal normalizers. {@code buildAligned()} yields
   * an {@link OffsetAwareNormalizer}: term identity is computed on normalized
   * text while every occurrence span stays in original text coordinates.
   */
  private static OffsetAwareNormalizer alignedNormalizer(
      List<PipelineOptions.NormalizerRung> rungs) {
    final TextNormalizer.Builder builder = TextNormalizer.builder();
    if (rungs.contains(PipelineOptions.NormalizerRung.STRIP_INVISIBLE)) {
      builder.stripInvisible();
    }
    if (rungs.contains(PipelineOptions.NormalizerRung.WHITESPACE)) {
      builder.whitespace();
    }
    if (rungs.contains(PipelineOptions.NormalizerRung.DASHES)) {
      builder.dashes();
    }
    if (rungs.contains(PipelineOptions.NormalizerRung.QUOTES)) {
      builder.quotes();
    }
    if (rungs.contains(PipelineOptions.NormalizerRung.DIGITS)) {
      builder.digits();
    }
    if (rungs.contains(PipelineOptions.NormalizerRung.FULL_CASE_FOLD)) {
      builder.fullCaseFold();
    }
    return builder.buildAligned();
  }

  /**
   * Wraps every stemmer in a {@link SharingStemmer}, which hands each thread
   * its own delegate from the factory: the classic {@code PorterStemmer} is
   * stateful, and even the stateless light stemmers stay cheap to duplicate.
   */
  private static Stemmer threadSafeStemmer(PipelineOptions.Stemmer choice) {
    final StemmerFactory factory = switch (choice) {
      case PORTER -> new PorterStemmerFactory();
      case SNOWBALL_ENGLISH -> () -> new SnowballStemmer(SnowballStemmer.ALGORITHM.ENGLISH);
      case SNOWBALL_GERMAN -> () -> new SnowballStemmer(SnowballStemmer.ALGORITHM.GERMAN);
      case SNOWBALL_FRENCH -> () -> new SnowballStemmer(SnowballStemmer.ALGORITHM.FRENCH);
      case SNOWBALL_SPANISH -> () -> new SnowballStemmer(SnowballStemmer.ALGORITHM.SPANISH);
      case LIGHT_ENGLISH -> new EnglishMinimalStemmer();
      case LIGHT_GERMAN -> new GermanLightStemmer();
      case LIGHT_FRENCH -> new FrenchLightStemmer();
      case LIGHT_SPANISH -> new SpanishLightStemmer();
      case NONE -> throw new IllegalArgumentException("NONE has no stemmer");
    };
    return new SharingStemmer(factory);
  }

  /** Serializes a model-based annotator whose implementation is not thread-safe. */
  private static final class SynchronizedAnnotator implements DocumentAnnotator {

    private final DocumentAnnotator delegate;

    private SynchronizedAnnotator(DocumentAnnotator delegate) {
      this.delegate = delegate;
    }

    @Override
    public synchronized Document annotate(Document document) {
      return delegate.annotate(document);
    }

    @Override
    public java.util.Set<LayerKey<?>> requires() {
      return delegate.requires();
    }

    @Override
    public java.util.Set<LayerKey<?>> provides() {
      return delegate.provides();
    }
  }
}
