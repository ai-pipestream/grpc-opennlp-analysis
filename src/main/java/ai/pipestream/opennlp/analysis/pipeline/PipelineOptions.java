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

import java.util.List;

import opennlp.tools.document.DocumentAnalyzer;

/**
 * Immutable, gRPC-free description of one analysis pipeline configuration.
 * Used as the cache key for shared {@link DocumentAnalyzer} instances: two
 * requests with equal options always share one pipeline.
 *
 * @param language ISO 639 language code, for example {@code "en"}
 * @param tokenizer tokenizer selection
 * @param sentenceDetection whether to split sentences
 * @param posTags whether POS tagging was requested
 * @param ner whether named-entity recognition was requested
 * @param lemmatize whether dictionary lemmatization was requested
 * @param stemmer stemmer selection
 * @param termVectors term vector specification, or {@code null} when disabled
 * @param embeddingSource embedding source layer, or {@code null} when disabled
 * @param noise whether noise scoring was requested
 * @param artifacts whether text artifact flagging was requested
 * @param glossary glossary matching specification, or {@code null} when disabled
 * @param pii whether PII extraction was requested
 * @param coref whether coreference resolution was requested
 * @param dependencyParse whether dependency parsing was requested
 * @param geo whether geocoding was requested
 * @param relations relation patterns to match, empty when disabled
 */
public record PipelineOptions(String language, Tokenizer tokenizer, boolean sentenceDetection,
                              boolean posTags, boolean ner, boolean lemmatize,
                              Stemmer stemmer,
                              TermVectorSpec termVectors, EmbeddingSource embeddingSource,
                              boolean noise, boolean artifacts,
                              GlossarySpec glossary, boolean pii, boolean coref,
                              boolean dependencyParse, boolean geo,
                              List<RelationPatternSpec> relations) {

  /** Canonicalizes the relations list: {@code null} becomes empty. */
  public PipelineOptions {
    relations = relations == null ? List.of() : List.copyOf(relations);
  }

  /**
   * Legacy convenience: same as the canonical constructor with every tier-1
   * addition (noise, artifacts, glossary, PII, coref, dependency parse,
   * relations) disabled.
   *
   * @param language ISO 639 language code
   * @param tokenizer tokenizer selection
   * @param sentenceDetection whether to split sentences
   * @param posTags whether POS tagging was requested
   * @param ner whether named-entity recognition was requested
   * @param lemmatize whether dictionary lemmatization was requested
   * @param stemmer stemmer selection
   * @param termVectors term vector specification, or {@code null}
   * @param embeddingSource embedding source layer, or {@code null}
   */
  public PipelineOptions(String language, Tokenizer tokenizer, boolean sentenceDetection,
                         boolean posTags, boolean ner, boolean lemmatize,
                         Stemmer stemmer,
                         TermVectorSpec termVectors, EmbeddingSource embeddingSource) {
    this(language, tokenizer, sentenceDetection, posTags, ner, lemmatize, stemmer,
        termVectors, embeddingSource, false, false, null, false, false, false, false,
        List.of());
  }

  /** Tokenizers. */
  public enum Tokenizer {
    /** Splits on whitespace only. */
    WHITESPACE,
    /** Character-class tokenizer: letters, digits, punctuation. */
    SIMPLE,
    /** Unicode UAX #29 word segmentation. */
    UAX29,
    /** Lattice tokenizer over a MeCab dictionary (CJK); needs a
     * server-configured dictionary directory (OPENNLP_LATTICE_DIC_DIR). */
    LATTICE,
    /** SentencePiece subword tokenizer; needs a server-configured model file
     * (OPENNLP_SENTENCEPIECE_MODEL). */
    SENTENCEPIECE
  }

  /**
   * Model-free stemmers. The {@code SNOWBALL_*} values map 1:1 onto
   * {@code SnowballStemmer.ALGORITHM}; the {@code LIGHT_*} and
   * {@code MINIMAL_*} values map onto the light stemmer classes;
   * {@code HUNSPELL} needs a server-configured dictionary. All stemmers work
   * on the raw token surface form: they neither lowercase nor fold.
   */
  public enum Stemmer {
    /** No stemming. */
    NONE,
    /** Classic Porter algorithm (English). */
    PORTER,
    /** Snowball English (Porter2). */
    SNOWBALL_ENGLISH,
    /** Snowball German. */
    SNOWBALL_GERMAN,
    /** Snowball French. */
    SNOWBALL_FRENCH,
    /** Snowball Spanish. */
    SNOWBALL_SPANISH,
    /** Snowball Arabic. */
    SNOWBALL_ARABIC,
    /** Snowball Catalan. */
    SNOWBALL_CATALAN,
    /** Snowball Danish. */
    SNOWBALL_DANISH,
    /** Snowball Dutch. */
    SNOWBALL_DUTCH,
    /** Snowball Finnish. */
    SNOWBALL_FINNISH,
    /** Snowball Greek. */
    SNOWBALL_GREEK,
    /** Snowball Hungarian. */
    SNOWBALL_HUNGARIAN,
    /** Snowball Indonesian. */
    SNOWBALL_INDONESIAN,
    /** Snowball Irish. */
    SNOWBALL_IRISH,
    /** Snowball Italian. */
    SNOWBALL_ITALIAN,
    /** Snowball Norwegian. */
    SNOWBALL_NORWEGIAN,
    /** The Snowball Porter program. */
    SNOWBALL_PORTER,
    /** Snowball Portuguese. */
    SNOWBALL_PORTUGUESE,
    /** Snowball Romanian. */
    SNOWBALL_ROMANIAN,
    /** Snowball Russian. */
    SNOWBALL_RUSSIAN,
    /** Snowball Swedish. */
    SNOWBALL_SWEDISH,
    /** Snowball Turkish. */
    SNOWBALL_TURKISH,
    /** English minimal stemmer. */
    LIGHT_ENGLISH,
    /** German light stemmer. */
    LIGHT_GERMAN,
    /** French light stemmer. */
    LIGHT_FRENCH,
    /** Spanish light stemmer. */
    LIGHT_SPANISH,
    /** Finnish light stemmer. */
    LIGHT_FINNISH,
    /** Hungarian light stemmer. */
    LIGHT_HUNGARIAN,
    /** Italian light stemmer. */
    LIGHT_ITALIAN,
    /** Norwegian light stemmer, Bokmal variety. */
    LIGHT_NORWEGIAN_BOKMAAL,
    /** Norwegian light stemmer, Nynorsk variety. */
    LIGHT_NORWEGIAN_NYNORSK,
    /** Portuguese light stemmer. */
    LIGHT_PORTUGUESE,
    /** Russian light stemmer. */
    LIGHT_RUSSIAN,
    /** Swedish light stemmer. */
    LIGHT_SWEDISH,
    /** German minimal stemmer. */
    MINIMAL_GERMAN,
    /** French minimal stemmer. */
    MINIMAL_FRENCH,
    /** Norwegian minimal stemmer, Bokmal variety. */
    MINIMAL_NORWEGIAN_BOKMAAL,
    /** Norwegian minimal stemmer, Nynorsk variety. */
    MINIMAL_NORWEGIAN_NYNORSK,
    /** Spanish minimal stemmer. */
    MINIMAL_SPANISH,
    /** Swedish minimal stemmer. */
    MINIMAL_SWEDISH,
    /** Hunspell dictionary stemmer (needs OPENNLP_HUNSPELL_AFF/DIC). */
    HUNSPELL
  }

  /** How much each term vector records. */
  public enum TermVectorMode {
    /** Frequency plus occurrence spans in original text coordinates. */
    FULL,
    /** Frequency only. */
    SCORING_ONLY
  }

  /** Which layer supplies term identity. */
  public enum TermVectorSource {
    /** Group by normalized token text; the steps define identity. */
    TOKENS,
    /** Group by stem; the stem alone defines identity, steps ignored. */
    STEMS,
    /** Group by the stem of the normalized token; steps run, then the stemmer. */
    NORMALIZED_STEMS
  }

  /** One step of the aligned normalizer chain. */
  public enum NormalizerStep {
    /** Removes invisible characters. */
    STRIP_INVISIBLE,
    /** Collapses whitespace runs to a single space. */
    WHITESPACE,
    /** Normalizes Unicode dashes to ASCII '-'. */
    DASHES,
    /** Normalizes Unicode quotes to ASCII quotes. */
    QUOTES,
    /** Normalizes Unicode digits to ASCII digits. */
    DIGITS,
    /** Unicode full case folding. */
    FULL_CASE_FOLD,
    /** Collapses ellipsis characters to "...". */
    ELLIPSIS,
    /** Normalizes Unicode bullets to '*'. */
    BULLETS,
    /** Converts emoji to emoticon text. */
    EMOJI_TO_EMOTICON,
    /** Converts emoticons to emoji. */
    EMOTICON_TO_EMOJI,
    /** Expands German umlauts and sharp-s to ASCII digraphs. */
    GERMAN_UMLAUT,
    /** Unicode NFKC compatibility normalization. */
    NFKC,
    /** Unicode NFC canonical composition. */
    NFC,
    /** Unicode confusable skeleton folding (UTS #39). */
    CONFUSABLE_SKELETON,
    /** Strips accents and diacritics, separate from case folding. */
    ACCENT_FOLD,
    /** Simple case folding, distinct from {@link #FULL_CASE_FOLD}. */
    CASE_FOLD,
    /** Collapses horizontal whitespace, keeps line structure. */
    LINE_BREAK_PRESERVING_WHITESPACE,
    /** Replaces URLs and mail addresses with a single space. */
    URL,
    /** Replaces runs of ASCII digits with a single space. */
    NUMBER,
    /** Replaces hashtags, handles, and RT markers with a single space. */
    SOCIAL_MEDIA,
    /** Collapses whitespace and shrinks repeated-character runs to two. */
    SHRINK;

    /**
     * Whether this step can report a character alignment and so compose into
     * the aligned whole-document chain. The JDK/regex-backed steps (Unicode
     * normalization, confusable/accent/case folding, and the URL/number/
     * social-media/shrink rewrites) cannot; they run per token instead.
     *
     * @return {@code true} when the step is offset-aware
     */
    public boolean isOffsetAware() {
      return switch (this) {
        case NFKC, NFC, CONFUSABLE_SKELETON, ACCENT_FOLD, CASE_FOLD, URL, NUMBER,
            SOCIAL_MEDIA, SHRINK -> false;
        default -> true;
      };
    }
  }

  /** Which annotation layer is embedded. */
  public enum EmbeddingSource {
    /** Embed each sentence (chunk embeddings). */
    SENTENCES,
    /** Embed each token. */
    TOKENS
  }

  /**
   * Glossary matching configuration: Aho-Corasick longest match of the given
   * entries over the raw text, reported with original-text spans.
   *
   * @param entries the glossary entries (id + surface term); aliases are
   *                multiple entries sharing one id. Never empty.
   * @param ignoreCase whether matching ignores case (per code point)
   */
  public record GlossarySpec(List<opennlp.tools.glossary.GlossaryEntry> entries,
                             boolean ignoreCase) {

    /** Validates and defensively copies the entries. */
    public GlossarySpec {
      if (entries == null || entries.isEmpty()) {
        throw new IllegalArgumentException("glossary entries must not be empty");
      }
      entries = List.copyOf(entries);
    }
  }

  /**
   * One relation extraction pattern: a dependency-path rule with a type.
   *
   * @param type relation type label attached to matches
   * @param path whitespace-separated dependency steps ({@code '<'} up,
   *             {@code '>'} down), for example {@code "<nsubj >obj"}
   * @param trigger optional lowercased pivot-token form, or {@code null}
   */
  public record RelationPatternSpec(String type, String path, String trigger) {

    /** Validates the mandatory fields. */
    public RelationPatternSpec {
      if (type == null || type.isBlank() || path == null || path.isBlank()) {
        throw new IllegalArgumentException("relation pattern type and path must not be blank");
      }
    }
  }

  /**
   * Term vector configuration.
   *
   * @param mode recording mode
   * @param steps normalizer steps for term identity; empty selects the server
   *              default ({@link #DEFAULT_STEPS}); ignored when
   *              {@code source} is {@link TermVectorSource#STEMS}
   * @param source term identity source
   */
  public record TermVectorSpec(TermVectorMode mode, List<NormalizerStep> steps,
                               TermVectorSource source) {

    /** Default steps: strip invisible, collapse whitespace, full case fold. */
    public static final List<NormalizerStep> DEFAULT_STEPS =
        List.of(NormalizerStep.STRIP_INVISIBLE, NormalizerStep.WHITESPACE,
            NormalizerStep.FULL_CASE_FOLD);

    /**
     * Canonicalizes the steps: empty resolves to {@link #DEFAULT_STEPS}, and
     * duplicates are removed and the order normalized, so equal step sets
     * produce equal keys (and equal normalizers) regardless of request order.
     *
     * @param mode recording mode
     * @param steps requested steps, possibly empty
     * @param source term identity source
     */
    public TermVectorSpec {
      steps = steps == null || steps.isEmpty()
          ? DEFAULT_STEPS : steps.stream().distinct().sorted().toList();
      source = source == null ? TermVectorSource.TOKENS : source;
    }

    /**
     * Token-source convenience: same as {@code TermVectorSpec(mode, steps, TOKENS)}.
     *
     * @param mode recording mode
     * @param steps requested steps, possibly empty
     */
    public TermVectorSpec(TermVectorMode mode, List<NormalizerStep> steps) {
      this(mode, steps, TermVectorSource.TOKENS);
    }
  }

  /**
   * Returns the default option-set: English, whitespace tokenizer, sentence
   * detection on, everything else off.
   *
   * @return the default options, never {@code null}
   */
  public static PipelineOptions defaults() {
    return new PipelineOptions("en", Tokenizer.WHITESPACE, true, false, false, false,
        Stemmer.NONE, null, null, false, false, null, false, false, false, false,
        List.of());
  }
}
