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

import static org.assertj.core.api.Assertions.assertThat;

import java.util.List;

import org.junit.jupiter.api.Test;

import opennlp.tools.document.Annotation;
import opennlp.tools.document.Document;
import opennlp.tools.termvector.TermVector;
import opennlp.tools.termvector.TermVectorAnnotator;
import opennlp.tools.util.Span;

/**
 * The additional aligned normalizer rungs: term identity follows the rungs,
 * occurrence spans stay in original text coordinates.
 */
class NormalizerRungTest {

  private static Document analyze(String text, PipelineOptions.NormalizerRung... rungs) {
    final PipelineOptions options = new PipelineOptions("en",
        PipelineOptions.Tokenizer.WHITESPACE, false, false, false, false,
        PipelineOptions.Stemmer.NONE,
        new PipelineOptions.TermVectorSpec(PipelineOptions.TermVectorMode.FULL,
            List.of(rungs)),
        null);
    return AnalysisPipeline.create(options, PipelineEnvironment.empty()).analyze(text);
  }

  @Test
  void germanUmlautRungExpandsAfterCaseFolding() {
    // FULL_CASE_FOLD applies before GERMAN_UMLAUT in the fixed rung order, so
    // "Müller" folds to "müller" and then expands to "mueller".
    final Document document = analyze("Müller Mueller",
        PipelineOptions.NormalizerRung.FULL_CASE_FOLD,
        PipelineOptions.NormalizerRung.GERMAN_UMLAUT);

    final List<Annotation<TermVector>> vectors =
        document.get(TermVectorAnnotator.TERM_VECTORS);
    assertThat(vectors).hasSize(1);
    assertThat(vectors.get(0).value().term()).isEqualTo("mueller");
    assertThat(vectors.get(0).value().frequency()).isEqualTo(2);
    assertThat(vectors.get(0).value().spans())
        .containsExactly(new Span(0, 6), new Span(7, 14));
  }

  @Test
  void ellipsisRungCollapsesEllipsisCharacters() {
    final Document document = analyze("a … b ...",
        PipelineOptions.NormalizerRung.WHITESPACE,
        PipelineOptions.NormalizerRung.ELLIPSIS);

    final List<Annotation<TermVector>> vectors =
        document.get(TermVectorAnnotator.TERM_VECTORS);
    // "…" and "..." both fold to "...", keeping their original spans.
    assertThat(vectors).extracting(a -> a.value().term())
        .containsExactlyInAnyOrder("a", "...", "b");
    assertThat(vectors).filteredOn(a -> a.value().term().equals("..."))
        .singleElement()
        .satisfies(a -> {
          assertThat(a.value().frequency()).isEqualTo(2);
          assertThat(a.value().spans())
              .containsExactly(new Span(2, 3), new Span(6, 9));
        });
  }

  @Test
  void digitsRungNormalizesUnicodeDigits() {
    final Document document = analyze("x４2 x42",
        PipelineOptions.NormalizerRung.WHITESPACE,
        PipelineOptions.NormalizerRung.DIGITS);

    final List<Annotation<TermVector>> vectors =
        document.get(TermVectorAnnotator.TERM_VECTORS);
    // Full-width ４ folds to ASCII 4, so both surface forms collapse.
    assertThat(vectors).hasSize(1);
    assertThat(vectors.get(0).value().term()).isEqualTo("x42");
    assertThat(vectors.get(0).value().frequency()).isEqualTo(2);
  }
}
