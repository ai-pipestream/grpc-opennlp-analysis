/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License. You may obtain a copy of the License at
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

import opennlp.tools.lemmatizer.Lemmatizer;
import opennlp.wordnet.MorphyLemmatizer;

/**
 * Adapts the WordNet {@link MorphyLemmatizer} to the tag inventory this
 * service actually sees. Morphy maps Penn Treebank prefixes ({@code N},
 * {@code V}, {@code J}, {@code R}); the deployed POS models emit Universal
 * Dependencies tags, which mostly coincide ({@code NOUN}, {@code VERB},
 * {@code ADJ}, {@code ADV}) but not always: {@code PROPN} maps to no part of
 * speech there, so every proper noun would come back unknown.
 *
 * <p>Two adaptations:</p>
 * <ul>
 *   <li>{@code PROPN} is rewritten to {@code NNP} so proper nouns lemmatize
 *       as nouns; every other tag passes through unchanged.</li>
 *   <li>A token whose tag is empty (no POS model configured) or maps to no
 *       part of speech is looked up across all four parts of speech, noun
 *       first, and the first validated lemma wins. A token with a real,
 *       mappable tag is lemmatized against that part of speech only.</li>
 * </ul>
 *
 * <p>Unknown words come back as {@code "O"}, the OpenNLP convention the
 * dictionary backend follows. Thread safety: the delegate is immutable and
 * the adapter holds no state.</p>
 */
public final class WordNetLemmatizer implements Lemmatizer {

  /** Fallback lookup order when no usable tag is available. */
  private static final List<String> ALL_POS = List.of("n", "v", "a", "r");

  private final MorphyLemmatizer delegate;

  /**
   * Wraps a Morphy lemmatizer.
   *
   * @param delegate the WordNet lemmatizer, must not be {@code null}
   */
  public WordNetLemmatizer(MorphyLemmatizer delegate) {
    if (delegate == null) {
      throw new IllegalArgumentException("delegate must not be null");
    }
    this.delegate = delegate;
  }

  /**
   * {@inheritDoc}
   *
   * <p>Tags are adapted as described on the class: {@code PROPN} becomes
   * {@code NNP}, and an empty or unmappable tag triggers the all-POS
   * fallback.</p>
   *
   * @throws IllegalArgumentException Thrown if {@code toks} or {@code tags} is
   *     {@code null}, contains a {@code null} element, or the two differ in
   *     length.
   */
  @Override
  public String[] lemmatize(String[] toks, String[] tags) {
    if (toks == null) {
      throw new IllegalArgumentException("Toks must not be null");
    }
    if (tags == null) {
      throw new IllegalArgumentException("Tags must not be null");
    }
    if (toks.length != tags.length) {
      throw new IllegalArgumentException("Toks and tags must have the same length, got "
          + toks.length + " and " + tags.length);
    }
    final String[] lemmas = new String[toks.length];
    for (int i = 0; i < toks.length; i++) {
      lemmas[i] = lemmaOf(toks[i], tags[i]);
    }
    return lemmas;
  }

  /**
   * {@inheritDoc}
   *
   * <p>Every returned list is a singleton: the adapter picks the most
   * preferred candidate, it does not enumerate.</p>
   *
   * @throws IllegalArgumentException Thrown if {@code toks} or {@code tags} is
   *     {@code null}, contains a {@code null} element, or the two differ in
   *     length.
   */
  @Override
  public List<List<String>> lemmatize(List<String> toks, List<String> tags) {
    if (toks == null || tags == null) {
      throw new IllegalArgumentException("Toks and tags must not be null");
    }
    final List<List<String>> lemmas = new ArrayList<>(toks.size());
    final String[] flat = lemmatize(
        toks.toArray(new String[0]), tags.toArray(new String[0]));
    for (final String lemma : flat) {
      lemmas.add(List.of(lemma));
    }
    return lemmas;
  }

  /**
   * Lemmatizes one token: the mapped tag first, then the all-POS fallback
   * when the tag names no part of speech.
   *
   * @param token the token, must not be {@code null}
   * @param tag the POS tag, must not be {@code null}
   * @return the lemma, or {@code "O"} when unknown
   */
  private String lemmaOf(String token, String tag) {
    if (token == null) {
      throw new IllegalArgumentException("Toks must not contain a null element");
    }
    if (tag == null) {
      throw new IllegalArgumentException("Tags must not contain a null element");
    }
    final String mapped = "PROPN".equals(tag) ? "NNP" : tag;
    final List<String> forTag = candidates(token, mapped);
    if (!forTag.isEmpty()) {
      return forTag.get(0);
    }
    if (!mapped.isEmpty() && mapsToPos(mapped)) {
      // A real, mappable tag is respected: unknown as that POS is unknown.
      return MorphyLemmatizer.UNKNOWN_LEMMA;
    }
    for (final String pos : ALL_POS) {
      final List<String> candidates = candidates(token, pos);
      if (!candidates.isEmpty()) {
        return candidates.get(0);
      }
    }
    return MorphyLemmatizer.UNKNOWN_LEMMA;
  }

  /**
   * @param token the token
   * @param tag the tag to look up under
   * @return the candidate lemmas, empty when unknown
   */
  private List<String> candidates(String token, String tag) {
    final List<String> candidates =
        delegate.lemmatize(List.of(token), List.of(tag)).get(0);
    return candidates.size() == 1
            && candidates.get(0).equals(MorphyLemmatizer.UNKNOWN_LEMMA)
        ? List.of() : candidates;
  }

  /**
   * Whether the tag maps to a WordNet part of speech under Morphy's
   * Penn-prefix mapping, without depending on its package-private helper:
   * empty does not, {@code ADJ}/{@code ADV} do, and otherwise the first
   * letter decides ({@code N}, {@code V}, {@code J}, {@code R}, one-letter
   * {@code A}/{@code S}).
   *
   * @param tag the tag, never {@code null}
   * @return {@code true} when the tag maps to a part of speech
   */
  private static boolean mapsToPos(String tag) {
    if (tag.isEmpty()) {
      return false;
    }
    final String upper = tag.toUpperCase(java.util.Locale.ROOT);
    if (upper.startsWith("ADJ") || upper.startsWith("ADV")) {
      return true;
    }
    return switch (upper.charAt(0)) {
      case 'N', 'V', 'J', 'R' -> true;
      case 'A', 'S' -> tag.length() == 1;
      default -> false;
    };
  }
}
