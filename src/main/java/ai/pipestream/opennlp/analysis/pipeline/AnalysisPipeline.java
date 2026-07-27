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
import java.util.Set;

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
import opennlp.tools.stemmer.hunspell.HunspellStemmerFactory;
import opennlp.tools.stemmer.light.EnglishMinimalStemmer;
import opennlp.tools.stemmer.light.FinnishLightStemmer;
import opennlp.tools.stemmer.light.FrenchLightStemmer;
import opennlp.tools.stemmer.light.FrenchMinimalStemmer;
import opennlp.tools.stemmer.light.GermanLightStemmer;
import opennlp.tools.stemmer.light.GermanMinimalStemmer;
import opennlp.tools.stemmer.light.HungarianLightStemmer;
import opennlp.tools.stemmer.light.ItalianLightStemmer;
import opennlp.tools.stemmer.light.NorwegianLightStemmer;
import opennlp.tools.stemmer.light.NorwegianMinimalStemmer;
import opennlp.tools.stemmer.light.NorwegianVariety;
import opennlp.tools.stemmer.light.PortugueseLightStemmer;
import opennlp.tools.stemmer.light.RussianLightStemmer;
import opennlp.tools.stemmer.light.SpanishLightStemmer;
import opennlp.tools.stemmer.light.SpanishMinimalStemmer;
import opennlp.tools.stemmer.light.SwedishLightStemmer;
import opennlp.tools.stemmer.light.SwedishMinimalStemmer;
import opennlp.tools.stemmer.snowball.SnowballStemmer;
import opennlp.tools.termvector.TermVectorAnnotator;
import opennlp.tools.tokenize.SimpleTokenizer;
import opennlp.tools.tokenize.WhitespaceTokenizer;
import opennlp.tools.util.normalizer.GermanUmlautCharSequenceNormalizer;
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
   * Requested features whose model or dictionary the environment lacks are
   * skipped and reported in {@link #warnings()}; the pipeline still serves
   * everything else.
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
      final Stemmer stemmer = stemmer(options.stemmer(), environment, warnings);
      if (stemmer != null) {
        builder.add(new StemmerAnnotator(stemmer));
      }
    }

    // Lemmatization is service-side (like stem-sourced term vectors):
    // LemmatizerAnnotator requires the POS layer, which needs a model, so the
    // pipeline only records whether the dictionary is missing; the service
    // joins the dictionary onto the tokens after analysis, using the POS
    // layer when it happens to exist and a neutral tag otherwise.
    if (options.lemmatize() && environment.lemmatizer() == null) {
      warnings.add("lemmatize was requested but no lemmatizer dictionary is "
          + "configured (OPENNLP_LEMMATIZER_DICT); lemmas stays empty");
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

    // Stem-sourced term vectors group occurrences service-side by the stems
    // layer; the TermVectorAnnotator only runs for token-sourced vectors,
    // where the aligned normalizer chain defines term identity.
    if (options.termVectors() != null
        && options.termVectors().source() == PipelineOptions.TermVectorSource.TOKENS) {
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
    if (rungs.contains(PipelineOptions.NormalizerRung.ELLIPSIS)) {
      builder.ellipsis();
    }
    if (rungs.contains(PipelineOptions.NormalizerRung.BULLETS)) {
      builder.bullets();
    }
    if (rungs.contains(PipelineOptions.NormalizerRung.EMOJI_TO_EMOTICON)) {
      builder.emojiToEmoticon();
    }
    if (rungs.contains(PipelineOptions.NormalizerRung.EMOTICON_TO_EMOJI)) {
      builder.emoticonToEmoji();
    }
    if (rungs.contains(PipelineOptions.NormalizerRung.FULL_CASE_FOLD)) {
      builder.fullCaseFold();
    }
    if (rungs.contains(PipelineOptions.NormalizerRung.GERMAN_UMLAUT)) {
      // No dedicated builder method, but the rung is offset-aware and composes.
      builder.with(GermanUmlautCharSequenceNormalizer.getInstance());
    }
    return builder.buildAligned();
  }

  /**
   * Resolves the stemmer for a pipeline, or {@code null} (with a warning) when
   * the choice needs a dictionary the environment lacks. Every stemmer is
   * wrapped in a {@link SharingStemmer}, which hands each thread its own
   * delegate from the factory: the classic {@code PorterStemmer} is stateful,
   * and even the stateless light stemmers stay cheap to duplicate.
   */
  private static Stemmer stemmer(PipelineOptions.Stemmer choice,
                                 PipelineEnvironment environment,
                                 List<String> warnings) {
    if (choice == PipelineOptions.Stemmer.HUNSPELL) {
      if (environment.hunspellDictionary() == null) {
        warnings.add("STEMMER_HUNSPELL was requested but no Hunspell dictionary is "
            + "configured (OPENNLP_HUNSPELL_AFF + OPENNLP_HUNSPELL_DIC); "
            + "stems stays empty");
        return null;
      }
      return new SharingStemmer(new HunspellStemmerFactory(environment.hunspellDictionary()));
    }
    return new SharingStemmer(stemmerFactory(choice));
  }

  /**
   * The factory behind each algorithmic stemmer. The {@code SNOWBALL_*} values
   * map 1:1 onto {@link SnowballStemmer.ALGORITHM} by name; the light and
   * minimal classes are their own factories.
   */
  private static StemmerFactory stemmerFactory(PipelineOptions.Stemmer choice) {
    final String name = choice.name();
    if (name.startsWith("SNOWBALL_")) {
      final SnowballStemmer.ALGORITHM algorithm =
          SnowballStemmer.ALGORITHM.valueOf(name.substring("SNOWBALL_".length()));
      return () -> new SnowballStemmer(algorithm);
    }
    return switch (choice) {
      case PORTER -> new PorterStemmerFactory();
      case LIGHT_ENGLISH -> new EnglishMinimalStemmer();
      case LIGHT_GERMAN -> new GermanLightStemmer();
      case LIGHT_FRENCH -> new FrenchLightStemmer();
      case LIGHT_SPANISH -> new SpanishLightStemmer();
      case LIGHT_FINNISH -> new FinnishLightStemmer();
      case LIGHT_HUNGARIAN -> new HungarianLightStemmer();
      case LIGHT_ITALIAN -> new ItalianLightStemmer();
      case LIGHT_NORWEGIAN_BOKMAAL -> new NorwegianLightStemmer(NorwegianVariety.BOKMAAL);
      case LIGHT_NORWEGIAN_NYNORSK -> new NorwegianLightStemmer(NorwegianVariety.NYNORSK);
      case LIGHT_PORTUGUESE -> new PortugueseLightStemmer();
      case LIGHT_RUSSIAN -> new RussianLightStemmer();
      case LIGHT_SWEDISH -> new SwedishLightStemmer();
      case MINIMAL_GERMAN -> new GermanMinimalStemmer();
      case MINIMAL_FRENCH -> new FrenchMinimalStemmer();
      case MINIMAL_NORWEGIAN_BOKMAAL -> new NorwegianMinimalStemmer(NorwegianVariety.BOKMAAL);
      case MINIMAL_NORWEGIAN_NYNORSK -> new NorwegianMinimalStemmer(NorwegianVariety.NYNORSK);
      case MINIMAL_SPANISH -> new SpanishMinimalStemmer();
      case MINIMAL_SWEDISH -> new SwedishMinimalStemmer();
      default -> throw new IllegalArgumentException("no factory for " + choice);
    };
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
    public Set<LayerKey<?>> requires() {
      return delegate.requires();
    }

    @Override
    public Set<LayerKey<?>> provides() {
      return delegate.provides();
    }
  }
}
