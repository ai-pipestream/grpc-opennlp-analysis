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
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import opennlp.tools.document.Annotation;
import opennlp.tools.document.Document;
import opennlp.tools.document.DocumentAnnotator;
import opennlp.tools.document.LayerKey;
import opennlp.tools.document.Layers;
import opennlp.tools.termvector.TermVector;
import opennlp.tools.termvector.TermVectorAnnotator;
import opennlp.tools.util.Span;
import opennlp.tools.util.normalizer.CharSequenceNormalizer;

/**
 * Term vector annotator for rung chains that cannot report a character
 * alignment. Rungs such as NFKC, confusable-skeleton folding, or accent/case
 * folding delegate to {@link java.text.Normalizer}, JDK case mapping, or
 * regexes and therefore cannot compose into the aligned whole-document chain
 * that {@link TermVectorAnnotator} requires.
 *
 * <p>This annotator normalizes each token's covered text on its own and
 * groups by the result — no alignment is needed, because occurrence spans are
 * the token layer's own spans and those are already in original text
 * coordinates. For the offset-aware rungs the two computations agree (token
 * text never contains the whitespace those rungs rewrite), so the choice of
 * annotator is invisible in the output: same layer key, same first-occurrence
 * order, same span semantics.</p>
 *
 * <p>Thread safety: stateless beyond the normalizer, which the rung
 * implementations keep thread-safe.</p>
 */
final class PerTokenTermVectorAnnotator implements DocumentAnnotator {

  private final CharSequenceNormalizer normalizer;
  private final TermVectorAnnotator.Mode mode;

  PerTokenTermVectorAnnotator(CharSequenceNormalizer normalizer,
                              TermVectorAnnotator.Mode mode) {
    if (normalizer == null || mode == null) {
      throw new IllegalArgumentException("normalizer and mode must not be null");
    }
    this.normalizer = normalizer;
    this.mode = mode;
  }

  @Override
  public Document annotate(Document document) {
    if (document == null) {
      throw new IllegalArgumentException("document must not be null");
    }
    if (!document.layers().contains(Layers.TOKENS)) {
      throw new IllegalArgumentException("document lacks the required layer "
          + Layers.TOKENS);
    }
    final CharSequence text = document.text();
    final Map<String, List<Span>> spansByTerm = new LinkedHashMap<>();
    for (Annotation<String> token : document.get(Layers.TOKENS)) {
      final String term =
          normalizer.normalize(token.span().getCoveredText(text)).toString();
      spansByTerm.computeIfAbsent(term, key -> new ArrayList<>()).add(token.span());
    }
    final List<Annotation<TermVector>> vectors = new ArrayList<>(spansByTerm.size());
    spansByTerm.forEach((term, spans) -> vectors.add(Annotation.of(
        mode == TermVectorAnnotator.Mode.FULL
            ? TermVector.withSpans(term, spans)
            : TermVector.count(term, spans.size()))));
    return document.with(TermVectorAnnotator.TERM_VECTORS, vectors);
  }

  @Override
  public Set<LayerKey<?>> requires() {
    return Set.of(Layers.TOKENS);
  }

  @Override
  public Set<LayerKey<?>> provides() {
    return Set.of(TermVectorAnnotator.TERM_VECTORS);
  }
}
