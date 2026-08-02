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

import java.util.Map;

import opennlp.tools.util.normalizer.CharSequenceNormalizer;

/**
 * Maps a symbol-joiner token onto its spelled-out word: {@code "&"} becomes
 * {@code "and"}. This is what lets a document writing "Dungeons &amp; Dragons"
 * and a query writing "dungeons and dragons" agree on term identity once terms
 * are folded — the ampersand token and the word "and" stem and fold to the
 * same term.
 *
 * <p>Only a token consisting of exactly the symbol is rewritten; an
 * ampersand embedded in a larger token ("R&amp;D", "AT&amp;T") is left alone,
 * because expanding it inside the token would invent a word that appears
 * nowhere. The table is deliberately one entry deep: add a symbol only when a
 * corpus shows the mismatch.</p>
 *
 * <p>Thread safety: stateless and immutable.</p>
 */
final class SymbolJoinerCharSequenceNormalizer implements CharSequenceNormalizer {

  private static final SymbolJoinerCharSequenceNormalizer INSTANCE =
      new SymbolJoinerCharSequenceNormalizer();

  private static final Map<String, String> WORD_BY_SYMBOL = Map.of("&", "and");

  private SymbolJoinerCharSequenceNormalizer() {
  }

  /**
   * @return the shared instance
   */
  static SymbolJoinerCharSequenceNormalizer getInstance() {
    return INSTANCE;
  }

  /**
   * Spells out a whole-token symbol joiner.
   *
   * @param text the token text, must not be {@code null}
   * @return the spelled-out word, or the input unchanged when the token is not
   *         exactly a known symbol
   */
  @Override
  public CharSequence normalize(CharSequence text) {
    if (text == null) {
      throw new IllegalArgumentException("text must not be null");
    }
    return WORD_BY_SYMBOL.getOrDefault(text.toString(), text.toString());
  }
}
