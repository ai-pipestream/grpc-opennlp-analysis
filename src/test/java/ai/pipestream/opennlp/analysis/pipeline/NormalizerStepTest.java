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
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import opennlp.tools.document.Annotation;
import opennlp.tools.document.Document;
import opennlp.tools.termvector.TermVector;
import opennlp.tools.termvector.TermVectorAnnotator;
import opennlp.tools.util.Span;

/**
 * The additional aligned normalizer steps: term identity follows the steps,
 * occurrence spans stay in original text coordinates.
 */
class NormalizerStepTest {

  private static Document analyze(String text, PipelineOptions.NormalizerStep... steps) {
    final PipelineOptions options = new PipelineOptions("en",
        PipelineOptions.Tokenizer.WHITESPACE, false, false, false, false,
        PipelineOptions.Stemmer.NONE,
        new PipelineOptions.TermVectorSpec(PipelineOptions.TermVectorMode.FULL,
            List.of(steps)),
        null);
    return AnalysisPipeline.create(options, PipelineEnvironment.empty()).analyze(text);
  }

  @Test
  void germanUmlautStepExpandsAfterCaseFolding() {
    // FULL_CASE_FOLD applies before GERMAN_UMLAUT in the fixed step order, so
    // "Müller" folds to "müller" and then expands to "mueller".
    final Document document = analyze("Müller Mueller",
        PipelineOptions.NormalizerStep.FULL_CASE_FOLD,
        PipelineOptions.NormalizerStep.GERMAN_UMLAUT);

    final List<Annotation<TermVector>> vectors =
        document.get(TermVectorAnnotator.TERM_VECTORS);
    assertThat(vectors).hasSize(1);
    assertThat(vectors.get(0).value().term()).isEqualTo("mueller");
    assertThat(vectors.get(0).value().frequency()).isEqualTo(2);
    assertThat(vectors.get(0).value().spans())
        .containsExactly(new Span(0, 6), new Span(7, 14));
  }

  @Test
  void ellipsisStepCollapsesEllipsisCharacters() {
    final Document document = analyze("a … b ...",
        PipelineOptions.NormalizerStep.WHITESPACE,
        PipelineOptions.NormalizerStep.ELLIPSIS);

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
  void symbolJoinerStepSpellsOutWholeTokenAmpersand() {
    // "Dungeons & Dragons" and a query typing "dungeons and dragons" agree on
    // term identity once "&" spells out; embedded ampersands stay untouched.
    final Document document = analyze("Dungeons & Dragons R&D",
        PipelineOptions.NormalizerStep.SYMBOL_JOINER,
        PipelineOptions.NormalizerStep.FULL_CASE_FOLD);

    final List<Annotation<TermVector>> vectors =
        document.get(TermVectorAnnotator.TERM_VECTORS);
    assertThat(vectors).extracting(a -> a.value().term())
        .containsExactlyInAnyOrder("dungeons", "and", "dragons", "r&d");
    // The "and" term keeps the ampersand's original-text span.
    assertThat(vectors).filteredOn(a -> a.value().term().equals("and"))
        .singleElement()
        .satisfies(a -> assertThat(a.value().spans())
            .containsExactly(new Span(9, 10)));
  }

  @Test
  void digitsStepNormalizesUnicodeDigits() {
    final Document document = analyze("x４2 x42",
        PipelineOptions.NormalizerStep.WHITESPACE,
        PipelineOptions.NormalizerStep.DIGITS);

    final List<Annotation<TermVector>> vectors =
        document.get(TermVectorAnnotator.TERM_VECTORS);
    // Full-width ４ folds to ASCII 4, so both surface forms collapse.
    assertThat(vectors).hasSize(1);
    assertThat(vectors.get(0).value().term()).isEqualTo("x42");
    assertThat(vectors.get(0).value().frequency()).isEqualTo(2);
  }

  @Test
  void nfkcStepFoldsLigatures() {
    // U+FB01 LATIN SMALL LIGATURE FI expands to "fi" under NFKC.
    final Document document = analyze("ﬁle file",
        PipelineOptions.NormalizerStep.NFKC);

    final List<Annotation<TermVector>> vectors =
        document.get(TermVectorAnnotator.TERM_VECTORS);
    assertThat(vectors).hasSize(1);
    assertThat(vectors.get(0).value().term()).isEqualTo("file");
    assertThat(vectors.get(0).value().frequency()).isEqualTo(2);
    // The ligature form is 3 UTF-16 units; the span stays in original
    // coordinates even though the normalized term is one char longer.
    assertThat(vectors.get(0).value().spans())
        .containsExactly(new Span(0, 3), new Span(4, 8));
  }

  @Test
  void nfcStepComposesAccentedCharacters() {
    // Composed é (U+00E9) and decomposed e + U+0301 collapse to one term.
    final Document document = analyze("café café",
        PipelineOptions.NormalizerStep.NFC);

    final List<Annotation<TermVector>> vectors =
        document.get(TermVectorAnnotator.TERM_VECTORS);
    assertThat(vectors).hasSize(1);
    assertThat(vectors.get(0).value().term()).isEqualTo("café");
    assertThat(vectors.get(0).value().spans())
        .containsExactly(new Span(0, 4), new Span(5, 10));
  }

  @Test
  void confusableSkeletonStepFoldsHomoglyphs() {
    // "paypal" spelled with a Cyrillic а (U+0430) collapses onto the Latin form.
    final Document document = analyze("pаypal paypal",
        PipelineOptions.NormalizerStep.CONFUSABLE_SKELETON);

    final List<Annotation<TermVector>> vectors =
        document.get(TermVectorAnnotator.TERM_VECTORS);
    assertThat(vectors).hasSize(1);
    assertThat(vectors.get(0).value().term()).isEqualTo("paypal");
    assertThat(vectors.get(0).value().frequency()).isEqualTo(2);
  }

  @Test
  void accentFoldStepStripsDiacriticsWithoutCaseFolding() {
    final Document document = analyze("Café cafe",
        PipelineOptions.NormalizerStep.ACCENT_FOLD);

    final List<Annotation<TermVector>> vectors =
        document.get(TermVectorAnnotator.TERM_VECTORS);
    // Accents fold but case does not: "Cafe" and "cafe" stay distinct.
    assertThat(vectors).extracting(a -> a.value().term())
        .containsExactly("Cafe", "cafe");
  }

  @Test
  void caseFoldStepLowercasesWithoutExpandingSharpS() {
    final Document document = analyze("COURT court Straße",
        PipelineOptions.NormalizerStep.CASE_FOLD);

    final List<Annotation<TermVector>> vectors =
        document.get(TermVectorAnnotator.TERM_VECTORS);
    assertThat(vectors).extracting(a -> a.value().term())
        .containsExactly("court", "straße");
    assertThat(vectors.get(0).value().frequency()).isEqualTo(2);
  }

  @Test
  void lineBreakPreservingWhitespaceStepKeepsOffsetsAcrossLineBreaks() {
    // The step is offset-aware and composes into the aligned chain; line
    // structure survives while occurrence spans stay exact.
    final Document document = analyze("one\n\n\ntwo two",
        PipelineOptions.NormalizerStep.LINE_BREAK_PRESERVING_WHITESPACE);

    final List<Annotation<TermVector>> vectors =
        document.get(TermVectorAnnotator.TERM_VECTORS);
    assertThat(vectors).extracting(a -> a.value().term())
        .containsExactly("one", "two");
    assertThat(vectors.get(1).value().spans())
        .containsExactly(new Span(6, 9), new Span(10, 13));
  }

  @Test
  void urlStepReplacesUrlsWithASpace() {
    final Document document = analyze("visit https://example.com/page now",
        PipelineOptions.NormalizerStep.URL);

    final List<Annotation<TermVector>> vectors =
        document.get(TermVectorAnnotator.TERM_VECTORS);
    assertThat(vectors).extracting(a -> a.value().term())
        .containsExactly("visit", " ", "now");
    // The space term keeps the original span of the whole URL.
    assertThat(vectors.get(1).value().spans())
        .containsExactly(new Span(6, 30));
  }

  @Test
  void numberStepReplacesDigitRunsWithASpace() {
    final Document document = analyze("order 12345 end",
        PipelineOptions.NormalizerStep.NUMBER);

    final List<Annotation<TermVector>> vectors =
        document.get(TermVectorAnnotator.TERM_VECTORS);
    assertThat(vectors).extracting(a -> a.value().term())
        .containsExactly("order", " ", "end");
  }

  @Test
  void socialMediaStepReplacesHashtagsAndHandles() {
    final Document document = analyze("#topic @user hello",
        PipelineOptions.NormalizerStep.SOCIAL_MEDIA);

    final List<Annotation<TermVector>> vectors =
        document.get(TermVectorAnnotator.TERM_VECTORS);
    // Both artifacts fold to a single space and group into one term.
    assertThat(vectors).extracting(a -> a.value().term())
        .containsExactly(" ", "hello");
    assertThat(vectors.get(0).value().frequency()).isEqualTo(2);
    assertThat(vectors.get(0).value().spans())
        .containsExactly(new Span(0, 6), new Span(7, 12));
  }

  @Test
  void shrinkStepCollapsesRepeatedCharacters() {
    final Document document = analyze("Helllllloooooo",
        PipelineOptions.NormalizerStep.SHRINK);

    final List<Annotation<TermVector>> vectors =
        document.get(TermVectorAnnotator.TERM_VECTORS);
    assertThat(vectors).hasSize(1);
    assertThat(vectors.get(0).value().term()).isEqualTo("Helloo");
  }

  @ParameterizedTest(name = "{0} builds and analyzes")
  @EnumSource(PipelineOptions.NormalizerStep.class)
  void everyStepBuildsAnOffsetAwarePipeline(PipelineOptions.NormalizerStep step) {
    final PipelineOptions options = new PipelineOptions("en",
        PipelineOptions.Tokenizer.WHITESPACE, false, false, false, false,
        PipelineOptions.Stemmer.NONE,
        new PipelineOptions.TermVectorSpec(PipelineOptions.TermVectorMode.FULL,
            List.of(step)),
        null);
    final Document document =
        AnalysisPipeline.create(options, PipelineEnvironment.empty())
            .analyze("Some TEXT with 12345 things");

    final List<Annotation<TermVector>> vectors =
        document.get(TermVectorAnnotator.TERM_VECTORS);
    assertThat(vectors).isNotEmpty();
    // Every occurrence span must map back inside the original text.
    assertThat(vectors).allSatisfy(a -> assertThat(a.value().spans())
        .allSatisfy(s -> assertThat(s.getEnd()).isLessThanOrEqualTo(29)));
  }
}
