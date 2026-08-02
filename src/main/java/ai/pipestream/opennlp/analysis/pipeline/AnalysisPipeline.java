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

import opennlp.geo.BundledGazetteer;
import opennlp.geo.PopulationPriorGeocoder;
import opennlp.spellcheck.normalizer.SpellCheckingCharSequenceNormalizer;
import opennlp.tools.artifacts.ArtifactAnnotator;
import opennlp.tools.assets.AssetAnnotator;
import opennlp.tools.coref.CorefAnnotator;
import opennlp.tools.depparse.DependencyAnnotator;
import opennlp.tools.depparse.FeedforwardDependencyParser;
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
import opennlp.tools.geo.DocumentRegionAnnotator;
import opennlp.tools.geo.GeocodeAnnotator;
import opennlp.tools.glossary.AhoCorasickGlossaryMatcher;
import opennlp.tools.glossary.GlossaryAnnotator;
import opennlp.tools.namefind.NameFinderME;
import opennlp.tools.noise.NoiseAnnotator;
import opennlp.tools.pii.CursorPiiExtractor;
import opennlp.tools.pii.PiiAnnotator;
import opennlp.tools.postag.POSTaggerME;
import opennlp.tools.relation.RelationAnnotator;
import opennlp.tools.relation.RelationPattern;
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
import opennlp.tools.tokenize.lattice.LatticeTokenizer;
import opennlp.tools.tokenize.uax29.WordTokenizer;
import opennlp.tools.util.normalizer.CharSequenceNormalizer;
import opennlp.tools.util.normalizer.ConfusableSkeletonCharSequenceNormalizer;
import opennlp.tools.util.normalizer.GermanUmlautCharSequenceNormalizer;
import opennlp.tools.util.normalizer.NumberCharSequenceNormalizer;
import opennlp.tools.util.normalizer.OffsetAwareNormalizer;
import opennlp.tools.util.normalizer.ShrinkCharSequenceNormalizer;
import opennlp.tools.util.normalizer.SymbolJoinerCharSequenceNormalizer;
import opennlp.tools.util.normalizer.SocialMediaCharSequenceNormalizer;
import opennlp.tools.util.normalizer.TextNormalizer;
import opennlp.tools.util.normalizer.UrlCharSequenceNormalizer;

/**
 * One compiled analysis pipeline for a distinct {@link PipelineOptions} set.
 * Holds a shared {@link DocumentAnalyzer} plus the metadata needed to read its
 * output back out. Construction is gRPC-free and unit-testable.
 *
 * <p>Thread safety: {@link DocumentAnalyzer}, {@code Document}, the aligned
 * normalizer, and the embedding model are thread-safe; stemmers are wrapped in
 * a {@link SharingStemmer} so each thread gets its own delegate; the
 * model-based POS tagger and name finder carry per-call adaptive state and are
 * therefore serialized behind a lock. The noise, artifact, glossary, and PII
 * annotators are stateless cursor scanners; the coref and relation annotators
 * document no per-call state. The dependency parser and the geo gazetteer/
 * geocoder have no documented thread-safety, so they are serialized behind a
 * lock defensively. A single instance serves concurrent
 * {@link #analyze(CharSequence)} calls.</p>
 */
public final class AnalysisPipeline {

  private final PipelineOptions options;
  private final DocumentAnalyzer analyzer;
  private final EmbeddingAnnotator embeddingAnnotator;
  private final Stemmer casedStemmer;
  private final CharSequenceNormalizer casedNormalizer;
  private final List<String> warnings;

  private AnalysisPipeline(PipelineOptions options, DocumentAnalyzer analyzer,
                           EmbeddingAnnotator embeddingAnnotator, Stemmer casedStemmer,
                           CharSequenceNormalizer casedNormalizer, List<String> warnings) {
    this.options = options;
    this.analyzer = analyzer;
    this.embeddingAnnotator = embeddingAnnotator;
    this.casedStemmer = casedStemmer;
    this.casedNormalizer = casedNormalizer;
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

    // Model availability gates for the model-consuming layers. A requested
    // feature whose prerequisite the environment lacks degrades to a warning,
    // never to a runtime model download. Computed first: the effective POS
    // decision below implies sentence detection, which is assembled early.
    final boolean canPos = environment.posModel() != null;
    final boolean canNer = environment.nerModel() != null;
    final boolean canDepparse = environment.depparseModel() != null;

    boolean dependencyParse = options.dependencyParse() || !options.relations().isEmpty();
    if (dependencyParse && !canDepparse) {
      warnings.add("dependency parsing was requested (directly or via relations) but "
          + "no dependency-parsing model is configured (OPENNLP_DEPPARSE_MODEL); "
          + "dependencies and relations stay empty");
      dependencyParse = false;
    } else if (dependencyParse && !canPos) {
      warnings.add("dependency parsing was requested (directly or via relations) but "
          + "no POS model is configured (OPENNLP_POS_MODEL); dependencies and "
          + "relations stay empty");
      dependencyParse = false;
    }

    boolean relations = !options.relations().isEmpty() && dependencyParse;
    if (!options.relations().isEmpty() && dependencyParse && !canNer) {
      warnings.add("relations were requested but no NER model is configured "
          + "(OPENNLP_NER_MODEL); relations stays empty");
      relations = false;
    }

    boolean coref = options.coref();
    if (coref && !canPos) {
      warnings.add("coref was requested but no POS model is configured "
          + "(OPENNLP_POS_MODEL); coref_mentions stays empty");
      coref = false;
    }
    if (coref && !canNer) {
      warnings.add("coref was requested but no NER model is configured "
          + "(OPENNLP_NER_MODEL); coref_mentions stays empty");
      coref = false;
    }

    boolean geo = options.geo();
    if (geo && !canNer) {
      warnings.add("geo was requested but no NER model is configured "
          + "(OPENNLP_NER_MODEL); locations and regions stay empty");
      geo = false;
    }

    // Implied layers: coref consumes POS+entities, relations consume
    // entities+dependencies, dependencies consume POS, geo consumes entities.
    // When the model exists the producing annotator runs even if the client
    // did not ask for the layer itself (the embeddings/sentence-detection
    // precedent), and the layers then surface in the response.
    final boolean posTags = options.posTags() || coref || dependencyParse;
    final boolean ner = options.ner() || coref || relations || geo;

    boolean sentenceDetection = options.sentenceDetection();
    if (options.embeddingSource() == PipelineOptions.EmbeddingSource.SENTENCES
        && !sentenceDetection) {
      sentenceDetection = true;
      warnings.add("embedding source SENTENCES requires sentence detection; "
          + "sentence_detection was enabled for this request");
    }
    if (posTags && canPos && !sentenceDetection) {
      // The POS annotator requires the sentences layer; without this a
      // pos_tags request with sentence_detection off would fail assembly.
      sentenceDetection = true;
      warnings.add("POS tagging (requested directly or implied by "
          + "coref/dependency_parse/relations) requires sentence detection; "
          + "sentence_detection was enabled for this request");
    }
    if (sentenceDetection) {
      // Newline-based and model-free: sentence offsets are always exact and the
      // default pipeline needs zero downloads.
      builder.add(new SentenceDetectorAnnotator(new NewlineSentenceDetector()));
    }

    builder.add(new TokenizerAnnotator(tokenizer(options.tokenizer(), environment, warnings)));

    if (options.noise()) {
      // Assets first: the noise scorer's default mode excludes embedded
      // assets (data URIs, base64 blobs) from scoring.
      builder.add(new AssetAnnotator());
      builder.add(new NoiseAnnotator());
    }

    if (options.artifacts()) {
      builder.add(new ArtifactAnnotator());
    }

    if (options.glossary() != null) {
      builder.add(new GlossaryAnnotator(new AhoCorasickGlossaryMatcher(
          options.glossary().entries(), options.glossary().ignoreCase())));
    }

    // Spell correction is a model-gated step: it joins the chain only when a
    // SymSpell model is configured, and warns instead when a request asks for
    // it without one. One normalizer instance serves every chain this
    // pipeline builds (folded, cased, and the stemmer decorator all agree on
    // what a corrected token is).
    SpellCheckingCharSequenceNormalizer spellcheck = null;
    if (options.termVectors() != null
        && options.termVectors().steps().contains(PipelineOptions.NormalizerStep.SPELLCHECK)) {
      if (environment.spellcheckModel() != null) {
        spellcheck = new SpellCheckingCharSequenceNormalizer(environment.spellcheckModel());
      } else {
        warnings.add("NORMALIZER_STEP_SPELLCHECK was requested but no spell-check "
            + "model is configured (OPENNLP_SPELLCHECK_MODEL); the step is skipped "
            + "for this pipeline");
      }
    }

    Stemmer baseStemmer = null;
    if (options.stemmer() != PipelineOptions.Stemmer.NONE) {
      baseStemmer = stemmer(options.stemmer(), environment, warnings);
      Stemmer stemmer = baseStemmer;
      // NORMALIZED_STEMS folds BEFORE stemming. Stemmers are written for
      // folded input, so without this "COURT" survives a Porter pass
      // unchanged while "court" stems normally, and one corpus ends up
      // holding both as distinct terms.
      if (stemmer != null
          && options.termVectors() != null
          && options.termVectors().source()
              == PipelineOptions.TermVectorSource.NORMALIZED_STEMS) {
        stemmer = normalizingStemmer(stemmer, options.termVectors().steps(), spellcheck);
      }
      if (stemmer != null) {
        builder.add(new StemmerAnnotator(stemmer));
      }
    }

    // Dual identity: the cased half is the same computation over the chain
    // minus the case-folding steps. It runs service-side, like the
    // stem-sourced grouping: the pipeline hands the service a stemmer over
    // the cased chain (NORMALIZED_STEMS) or the cased normalizer itself
    // (TOKENS), and the service groups occurrences under both identities in
    // one pass.
    Stemmer casedStemmer = null;
    CharSequenceNormalizer casedNormalizer = null;
    final PipelineOptions.TermVectorSpec dualSpec = options.termVectors();
    if (dualSpec != null && dualSpec.dualCased()) {
      if (dualSpec.casedSteps().equals(dualSpec.steps())) {
        warnings.add("dual_cased was requested but the normalizer step chain has "
            + "no case-folding step; cased_term_vectors will duplicate term_vectors");
      }
      if (dualSpec.source() == PipelineOptions.TermVectorSource.NORMALIZED_STEMS) {
        // baseStemmer is null here when the dictionary is missing; that case
        // already warned above, and cased_term_vectors stays as empty as stems.
        if (baseStemmer != null) {
          casedStemmer = normalizingStemmer(baseStemmer, dualSpec.casedSteps(), spellcheck);
        }
      } else {
        casedNormalizer = normalizerChain(dualSpec.casedSteps(), spellcheck);
      }
    }

    // Lemmatization is service-side (like stem-sourced term vectors):
    // LemmatizerAnnotator requires the POS layer, which needs a model, so the
    // pipeline only records whether a backend is missing; the service joins
    // the backend (WordNet Morphy or the flat dictionary) onto the tokens
    // after analysis, using the POS layer when it happens to exist and the
    // backend's tagless path otherwise.
    if (options.lemmatize() && environment.lemmatizer() == null) {
      warnings.add("lemmatize was requested but no lemmatizer backend is "
          + "configured (OPENNLP_WORDNET_DIR or OPENNLP_LEMMATIZER_DICT); "
          + "lemmas stays empty");
    }

    if (options.pii()) {
      builder.add(new PiiAnnotator(new CursorPiiExtractor()));
    }

    if (posTags) {
      if (canPos) {
        // Guarded: the tagger keeps per-call adaptive state in its searcher.
        builder.add(new SynchronizedAnnotator(
            new POSTaggerAnnotator(new POSTaggerME(environment.posModel()))));
      } else if (options.posTags()) {
        warnings.add("pos_tags was requested but no POS model is configured "
            + "(OPENNLP_POS_MODEL); pos stays empty");
      }
    }

    if (ner) {
      if (canNer) {
        // Guarded: the name finder clears adaptive data per call.
        builder.add(new SynchronizedAnnotator(
            new NameFinderAnnotator(new NameFinderME(environment.nerModel()))));
      } else if (options.ner()) {
        warnings.add("ner was requested but no NER model is configured "
            + "(OPENNLP_NER_MODEL); entities stays empty");
      }
    }

    if (dependencyParse) {
      // Guarded: parser thread-safety is not documented as shareable.
      builder.add(new SynchronizedAnnotator(new DependencyAnnotator(
          new FeedforwardDependencyParser(environment.depparseModel()))));
    }

    if (coref) {
      builder.add(new CorefAnnotator());
    }

    if (relations) {
      builder.add(new RelationAnnotator(options.relations().stream()
          .map(p -> new RelationPattern(p.type(), p.path(), p.trigger()))
          .toList()));
    }

    if (geo) {
      // Guarded: gazetteer/geocoder thread-safety is unverified. The bundled
      // Natural Earth gazetteer needs no external files.
      builder.add(new SynchronizedAnnotator(new GeocodeAnnotator(
          new PopulationPriorGeocoder(BundledGazetteer.getInstance()))));
      builder.add(new SynchronizedAnnotator(new DocumentRegionAnnotator()));
    }

    // Stem-sourced term vectors group occurrences service-side by the stems
    // layer. For token-sourced vectors the aligned TermVectorAnnotator runs
    // when every step can report an alignment; a chain containing a
    // JDK/regex-backed step (NFKC, confusable skeleton, accent/case fold,
    // URL/number/social/shrink) goes through the per-token annotator, which
    // needs no alignment because occurrence spans are the token spans.
    if (options.termVectors() != null
        && options.termVectors().source() == PipelineOptions.TermVectorSource.TOKENS) {
      final PipelineOptions.TermVectorSpec spec = options.termVectors();
      final TermVectorAnnotator.Mode mode =
          spec.mode() == PipelineOptions.TermVectorMode.SCORING_ONLY
              ? TermVectorAnnotator.Mode.SCORING_ONLY : TermVectorAnnotator.Mode.FULL;
      if (spec.steps().stream().allMatch(PipelineOptions.NormalizerStep::isOffsetAware)) {
        builder.add(new TermVectorAnnotator(alignedNormalizer(spec.steps()), mode));
      } else {
        builder.add(new PerTokenTermVectorAnnotator(normalizerChain(spec.steps(), spellcheck), mode));
      }
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
        casedStemmer, casedNormalizer, List.copyOf(warnings));
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
   * @return the stemmer over the case-preserving step chain for a
   *         dual-identity request, or {@code null} when dual identity was not
   *         requested, the source is not NORMALIZED_STEMS, or the stemmer's
   *         dictionary is missing
   */
  public Stemmer casedStemmer() {
    return casedStemmer;
  }

  /**
   * @return the case-preserving normalizer chain for a dual-identity request
   *         over the TOKENS source, or {@code null} when dual identity was not
   *         requested or the source stems
   */
  public CharSequenceNormalizer casedNormalizer() {
    return casedNormalizer;
  }

  /**
   * @return requested-but-unavailable features, surfaced on every response
   *         produced through this pipeline; never {@code null}
   */
  public List<String> warnings() {
    return warnings;
  }

  /**
   * Builds the aligned normalizer for the requested steps. Steps apply in the
   * fixed order of the OpenNLP builder, independent of request order, so equal
   * step sets always produce equal normalizers. {@code buildAligned()} yields
   * an {@link OffsetAwareNormalizer}: term identity is computed on normalized
   * text while every occurrence span stays in original text coordinates.
   */
  /**
   * Wraps a stemmer so the normalizer step chain runs on each token before
   * it is stemmed, making the stem of the normalized form the term
   * identity.
   *
   * <p>This is deliberately the same step list the token path would use, so
   * NORMALIZED_STEMS and TOKENS agree on what "the same word" means and
   * differ only in whether the stemmer then applies.
   */
  private static Stemmer normalizingStemmer(
      Stemmer delegate, List<PipelineOptions.NormalizerStep> steps,
      CharSequenceNormalizer spellcheck) {
    final CharSequenceNormalizer normalizer = normalizerChain(steps, spellcheck);
    return text -> delegate.stem(normalizer.normalize(text));
  }

  /** The step chain as a plain normalizer (no offset alignment needed: the
   * token's offsets come from the tokenizer, not from its stemmed form). */
  private static CharSequenceNormalizer normalizerChain(
      List<PipelineOptions.NormalizerStep> steps, CharSequenceNormalizer spellcheck) {
    return normalizerBuilder(steps, spellcheck).build();
  }

  private static OffsetAwareNormalizer alignedNormalizer(
      List<PipelineOptions.NormalizerStep> steps) {
    return normalizerBuilder(steps, null).buildAligned();
  }

  /** Populates a normalizer builder from the step list; the two build
   * methods (aligned for term vectors, plain for the stemmer decorator)
   * must see the SAME chain or the two sources would disagree on
   * identity. {@code spellcheck} is the model-gated correction step, or
   * {@code null} when unavailable or unrequested; it is never part of an
   * aligned chain (it cannot report an alignment, so a chain containing it
   * always builds plain). */
  private static TextNormalizer.Builder normalizerBuilder(
      List<PipelineOptions.NormalizerStep> steps, CharSequenceNormalizer spellcheck) {
    final TextNormalizer.Builder builder = TextNormalizer.builder();
    // Compatibility normalization first: it rewrites the characters every
    // later step matches on (full-width forms, ligatures, …).
    if (steps.contains(PipelineOptions.NormalizerStep.NFKC)) {
      builder.nfkc();
    }
    if (steps.contains(PipelineOptions.NormalizerStep.NFC)) {
      builder.nfc();
    }
    if (steps.contains(PipelineOptions.NormalizerStep.STRIP_INVISIBLE)) {
      builder.stripInvisible();
    }
    if (steps.contains(PipelineOptions.NormalizerStep.WHITESPACE)) {
      builder.whitespace();
    }
    if (steps.contains(PipelineOptions.NormalizerStep.LINE_BREAK_PRESERVING_WHITESPACE)) {
      builder.whitespacePreservingLineBreaks();
    }
    if (steps.contains(PipelineOptions.NormalizerStep.DASHES)) {
      builder.dashes();
    }
    if (steps.contains(PipelineOptions.NormalizerStep.QUOTES)) {
      builder.quotes();
    }
    if (steps.contains(PipelineOptions.NormalizerStep.DIGITS)) {
      builder.digits();
    }
    if (steps.contains(PipelineOptions.NormalizerStep.URL)) {
      builder.with(UrlCharSequenceNormalizer.getInstance());
    }
    if (steps.contains(PipelineOptions.NormalizerStep.NUMBER)) {
      builder.with(NumberCharSequenceNormalizer.getInstance());
    }
    if (steps.contains(PipelineOptions.NormalizerStep.SOCIAL_MEDIA)) {
      builder.with(SocialMediaCharSequenceNormalizer.getInstance());
    }
    if (steps.contains(PipelineOptions.NormalizerStep.SHRINK)) {
      builder.with(ShrinkCharSequenceNormalizer.getInstance());
    }
    if (steps.contains(PipelineOptions.NormalizerStep.SYMBOL_JOINER)) {
      // Whole-token symbol expansion ("&" -> "and"); runs before the folding
      // steps so the spelled-out word folds and stems like any other token.
      builder.with(SymbolJoinerCharSequenceNormalizer.getInstance());
    }
    if (spellcheck != null
        && steps.contains(PipelineOptions.NormalizerStep.SPELLCHECK)) {
      // After the noise-removing rewrites, before the folds: the corrector
      // sees clean tokens and re-applies their casing pattern, which the
      // folding steps then treat like any other casing.
      builder.with(spellcheck);
    }
    if (steps.contains(PipelineOptions.NormalizerStep.CONFUSABLE_SKELETON)) {
      builder.with(ConfusableSkeletonCharSequenceNormalizer.getInstance());
    }
    if (steps.contains(PipelineOptions.NormalizerStep.ACCENT_FOLD)) {
      builder.accentFold();
    }
    if (steps.contains(PipelineOptions.NormalizerStep.CASE_FOLD)) {
      builder.caseFold();
    }
    if (steps.contains(PipelineOptions.NormalizerStep.ELLIPSIS)) {
      builder.ellipsis();
    }
    if (steps.contains(PipelineOptions.NormalizerStep.BULLETS)) {
      builder.bullets();
    }
    if (steps.contains(PipelineOptions.NormalizerStep.EMOJI_TO_EMOTICON)) {
      builder.emojiToEmoticon();
    }
    if (steps.contains(PipelineOptions.NormalizerStep.EMOTICON_TO_EMOJI)) {
      builder.emoticonToEmoji();
    }
    if (steps.contains(PipelineOptions.NormalizerStep.FULL_CASE_FOLD)) {
      builder.fullCaseFold();
    }
    if (steps.contains(PipelineOptions.NormalizerStep.GERMAN_UMLAUT)) {
      // No dedicated builder method, but the step is offset-aware and composes.
      builder.with(GermanUmlautCharSequenceNormalizer.getInstance());
    }
    return builder;
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

  /**
   * Resolves the tokenizer for a pipeline. UAX29 is model-free; the lattice
   * and SentencePiece tokenizers need a server-configured dictionary/model —
   * when the environment lacks it, the request is served with the whitespace
   * tokenizer and a loud warning (a tokenizer is not optional; the pipeline
   * must produce tokens).
   */
  private static opennlp.tools.tokenize.Tokenizer tokenizer(
      PipelineOptions.Tokenizer choice, PipelineEnvironment environment,
      List<String> warnings) {
    return switch (choice) {
      case WHITESPACE -> WhitespaceTokenizer.INSTANCE;
      case SIMPLE -> SimpleTokenizer.INSTANCE;
      case UAX29 -> new WordTokenizer();
      case LATTICE -> {
        if (environment.mecabDictionary() != null) {
          yield new LatticeTokenizer(environment.mecabDictionary());
        }
        warnings.add("TOKENIZER_LATTICE was requested but no MeCab dictionary is "
            + "configured (OPENNLP_LATTICE_DIC_DIR); falling back to whitespace "
            + "tokenization for this pipeline");
        yield WhitespaceTokenizer.INSTANCE;
      }
      case SENTENCEPIECE -> {
        if (environment.sentencePieceTokenizer() != null) {
          yield new SentencePieceTokenizerAdapter(environment.sentencePieceTokenizer());
        }
        warnings.add("TOKENIZER_SENTENCEPIECE was requested but no SentencePiece model "
            + "is configured (OPENNLP_SENTENCEPIECE_MODEL); falling back to whitespace "
            + "tokenization for this pipeline");
        yield WhitespaceTokenizer.INSTANCE;
      }
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
