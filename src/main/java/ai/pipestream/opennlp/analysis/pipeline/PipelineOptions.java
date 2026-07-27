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
 * @param stemmer stemmer selection
 * @param termVectors term vector specification, or {@code null} when disabled
 * @param embeddingSource embedding source layer, or {@code null} when disabled
 */
public record PipelineOptions(String language, Tokenizer tokenizer, boolean sentenceDetection,
                              boolean posTags, boolean ner, Stemmer stemmer,
                              TermVectorSpec termVectors, EmbeddingSource embeddingSource) {

  /** Model-free tokenizers. */
  public enum Tokenizer {
    /** Splits on whitespace only. */
    WHITESPACE,
    /** Character-class tokenizer: letters, digits, punctuation. */
    SIMPLE
  }

  /** Model-free, algorithmic stemmers. */
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
    /** English minimal stemmer. */
    LIGHT_ENGLISH,
    /** German light stemmer. */
    LIGHT_GERMAN,
    /** French light stemmer. */
    LIGHT_FRENCH,
    /** Spanish light stemmer. */
    LIGHT_SPANISH
  }

  /** How much each term vector records. */
  public enum TermVectorMode {
    /** Frequency plus occurrence spans in original text coordinates. */
    FULL,
    /** Frequency only. */
    SCORING_ONLY
  }

  /** One rung of the aligned normalizer chain. */
  public enum NormalizerRung {
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
    FULL_CASE_FOLD
  }

  /** Which annotation layer is embedded. */
  public enum EmbeddingSource {
    /** Embed each sentence (chunk embeddings). */
    SENTENCES,
    /** Embed each token. */
    TOKENS
  }

  /**
   * Term vector configuration.
   *
   * @param mode recording mode
   * @param rungs normalizer rungs for term identity; empty selects the server
   *              default ({@link #DEFAULT_RUNGS})
   */
  public record TermVectorSpec(TermVectorMode mode, List<NormalizerRung> rungs) {

    /** Default rungs: strip invisible, collapse whitespace, full case fold. */
    public static final List<NormalizerRung> DEFAULT_RUNGS =
        List.of(NormalizerRung.STRIP_INVISIBLE, NormalizerRung.WHITESPACE,
            NormalizerRung.FULL_CASE_FOLD);

    /**
     * Canonicalizes the rungs: empty resolves to {@link #DEFAULT_RUNGS}, and
     * duplicates are removed and the order normalized, so equal rung sets
     * produce equal keys (and equal normalizers) regardless of request order.
     *
     * @param mode recording mode
     * @param rungs requested rungs, possibly empty
     */
    public TermVectorSpec {
      rungs = rungs == null || rungs.isEmpty()
          ? DEFAULT_RUNGS : rungs.stream().distinct().sorted().toList();
    }
  }

  /**
   * Returns the default option-set: English, whitespace tokenizer, sentence
   * detection on, everything else off.
   *
   * @return the default options, never {@code null}
   */
  public static PipelineOptions defaults() {
    return new PipelineOptions("en", Tokenizer.WHITESPACE, true, false, false,
        Stemmer.NONE, null, null);
  }
}
