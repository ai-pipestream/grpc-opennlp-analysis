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
import opennlp.tools.document.Layers;
import opennlp.tools.stemmer.StemmerAnnotator;
import opennlp.tools.termvector.TermVector;
import opennlp.tools.termvector.TermVectorAnnotator;
import opennlp.tools.util.Span;

/**
 * Pipeline construction and term-vector correctness through the full aligned
 * normalizer chain — the offset-fidelity guarantee the service is built on.
 */
class AnalysisPipelineTest {

  private static final String CASE_FOLD_SAMPLE = "Groß  groß  GROSS";

  private static AnalysisPipeline create(PipelineOptions options) {
    return AnalysisPipeline.create(options, PipelineEnvironment.empty());
  }

  @Test
  void fullTermVectorsKeepOriginalOffsetsThroughCaseFolding() {
    final PipelineOptions options = new PipelineOptions("en",
        PipelineOptions.Tokenizer.WHITESPACE, false, false, false, false,
        PipelineOptions.Stemmer.NONE,
        new PipelineOptions.TermVectorSpec(PipelineOptions.TermVectorMode.FULL,
            List.of(PipelineOptions.NormalizerRung.WHITESPACE,
                PipelineOptions.NormalizerRung.FULL_CASE_FOLD)),
        null);

    final Document document = create(options).analyze(CASE_FOLD_SAMPLE);

    final List<Annotation<TermVector>> vectors =
        document.get(TermVectorAnnotator.TERM_VECTORS);
    assertThat(vectors).hasSize(1);
    final TermVector vector = vectors.get(0).value();
    // Full case folding collapses all three surface forms into one term.
    assertThat(vector.term()).isEqualTo("gross");
    assertThat(vector.frequency()).isEqualTo(3);
    // Occurrences point into the ORIGINAL text, double spaces included.
    assertThat(vector.spans()).containsExactly(
        new Span(0, 4), new Span(6, 10), new Span(12, 17));
    assertThat(vector.spans().get(0).getCoveredText(CASE_FOLD_SAMPLE).toString())
        .isEqualTo("Groß");
    assertThat(vector.spans().get(2).getCoveredText(CASE_FOLD_SAMPLE).toString())
        .isEqualTo("GROSS");
  }

  @Test
  void scoringOnlyRecordsFrequenciesWithoutSpans() {
    final PipelineOptions options = new PipelineOptions("en",
        PipelineOptions.Tokenizer.WHITESPACE, false, false, false, false,
        PipelineOptions.Stemmer.NONE,
        new PipelineOptions.TermVectorSpec(PipelineOptions.TermVectorMode.SCORING_ONLY,
            List.of(PipelineOptions.NormalizerRung.WHITESPACE,
                PipelineOptions.NormalizerRung.FULL_CASE_FOLD)),
        null);

    final Document document = create(options).analyze(CASE_FOLD_SAMPLE);

    final List<Annotation<TermVector>> vectors =
        document.get(TermVectorAnnotator.TERM_VECTORS);
    assertThat(vectors).hasSize(1);
    assertThat(vectors.get(0).value().term()).isEqualTo("gross");
    assertThat(vectors.get(0).value().frequency()).isEqualTo(3);
    assertThat(vectors.get(0).value().spans()).isNullOrEmpty();
  }

  @Test
  void defaultRungsApplyWhenNoneAreRequested() {
    final PipelineOptions options = new PipelineOptions("en",
        PipelineOptions.Tokenizer.WHITESPACE, false, false, false, false,
        PipelineOptions.Stemmer.NONE,
        new PipelineOptions.TermVectorSpec(PipelineOptions.TermVectorMode.FULL,
            List.of()),
        null);

    final Document document = create(options).analyze("Groß groß");

    final List<Annotation<TermVector>> vectors =
        document.get(TermVectorAnnotator.TERM_VECTORS);
    // Default rungs include full case folding, so both forms collapse.
    assertThat(vectors).hasSize(1);
    assertThat(vectors.get(0).value().term()).isEqualTo("gross");
    assertThat(vectors.get(0).value().frequency()).isEqualTo(2);
  }

  @Test
  void simpleTokenizerSplitsPunctuationIntoOwnTokens() {
    final PipelineOptions options = new PipelineOptions("en",
        PipelineOptions.Tokenizer.SIMPLE, false, false, false, false,
        PipelineOptions.Stemmer.NONE, null, null);

    final Document document = create(options).analyze("Hello, world!");

    assertThat(document.get(Layers.TOKENS))
        .extracting(a -> a.span().getCoveredText("Hello, world!").toString())
        .containsExactly("Hello", ",", "world", "!");
  }

  @Test
  void newlineSentenceDetectionProducesExactSpans() {
    final String text = "One two.\nThree four.";
    final PipelineOptions options = new PipelineOptions("en",
        PipelineOptions.Tokenizer.WHITESPACE, true, false, false, false,
        PipelineOptions.Stemmer.NONE, null, null);

    final Document document = create(options).analyze(text);

    final List<Annotation<String>> sentences = document.get(Layers.SENTENCES);
    assertThat(sentences).hasSize(2);
    assertThat(sentences.get(0).span().getCoveredText(text).toString()).isEqualTo("One two.");
    assertThat(sentences.get(1).span().getCoveredText(text).toString())
        .isEqualTo("Three four.");
  }

  @Test
  void porterStemsAreParallelToTokens() {
    final PipelineOptions options = new PipelineOptions("en",
        PipelineOptions.Tokenizer.WHITESPACE, false, false, false, false,
        PipelineOptions.Stemmer.PORTER, null, null);

    final Document document = create(options).analyze("running runs");

    final List<Annotation<String>> stems = document.get(StemmerAnnotator.STEMS);
    assertThat(stems).hasSameSizeAs(document.get(Layers.TOKENS));
    assertThat(stems).extracting(Annotation::value).containsExactly("run", "run");
  }

  @Test
  void lightGermanStemmerIsModelFree() {
    final PipelineOptions options = new PipelineOptions("de",
        PipelineOptions.Tokenizer.WHITESPACE, false, false, false, false,
        PipelineOptions.Stemmer.LIGHT_GERMAN, null, null);

    final Document document = create(options).analyze("Häuser Hause");

    // Light stemmers do not lowercase; "Häuser" and "Hause" both stem to "Haus".
    assertThat(document.get(StemmerAnnotator.STEMS))
        .extracting(Annotation::value)
        .containsExactly("Haus", "Haus");
  }

  @Test
  void posRequestWithoutModelWarnsAndStillAnalyzes() {
    final PipelineOptions options = new PipelineOptions("en",
        PipelineOptions.Tokenizer.WHITESPACE, false, true, false, false,
        PipelineOptions.Stemmer.NONE, null, null);

    final AnalysisPipeline pipeline = create(options);

    assertThat(pipeline.warnings()).anySatisfy(w -> assertThat(w).contains("POS"));
    assertThat(pipeline.analyze("hello world").get(Layers.TOKENS)).hasSize(2);
  }

  @Test
  void nerRequestWithoutModelWarnsAndStillAnalyzes() {
    final PipelineOptions options = new PipelineOptions("en",
        PipelineOptions.Tokenizer.WHITESPACE, false, false, true, false,
        PipelineOptions.Stemmer.NONE, null, null);

    final AnalysisPipeline pipeline = create(options);

    assertThat(pipeline.warnings()).anySatisfy(w -> assertThat(w).contains("NER"));
    assertThat(pipeline.analyze("hello world").get(Layers.TOKENS)).hasSize(2);
  }

  @Test
  void embeddingRequestWithoutModelWarnsAndStillAnalyzes() {
    final PipelineOptions options = new PipelineOptions("en",
        PipelineOptions.Tokenizer.WHITESPACE, true, false, false, false,
        PipelineOptions.Stemmer.NONE, null, PipelineOptions.EmbeddingSource.SENTENCES);

    final AnalysisPipeline pipeline = create(options);

    assertThat(pipeline.warnings()).anySatisfy(w -> assertThat(w).contains("embedding"));
    assertThat(pipeline.embeddingLayer()).isNull();
    assertThat(pipeline.analyze("one\ntwo").get(Layers.SENTENCES)).hasSize(2);
  }

  @Test
  void sentenceEmbeddingsForceSentenceDetectionOn() {
    final PipelineOptions options = new PipelineOptions("en",
        PipelineOptions.Tokenizer.WHITESPACE, false, false, false, false,
        PipelineOptions.Stemmer.NONE, null, PipelineOptions.EmbeddingSource.SENTENCES);

    final AnalysisPipeline pipeline = create(options);

    assertThat(pipeline.warnings())
        .anySatisfy(w -> assertThat(w).contains("sentence detection"));
    assertThat(pipeline.analyze("one\ntwo").get(Layers.SENTENCES)).hasSize(2);
  }

  @Test
  void equalOptionSetsProduceCacheableKeys() {
    final PipelineOptions a = new PipelineOptions("en",
        PipelineOptions.Tokenizer.WHITESPACE, true, false, false, false,
        PipelineOptions.Stemmer.PORTER,
        new PipelineOptions.TermVectorSpec(PipelineOptions.TermVectorMode.FULL,
            List.of(PipelineOptions.NormalizerRung.WHITESPACE,
                PipelineOptions.NormalizerRung.FULL_CASE_FOLD)),
        null);
    // Same rungs in a different order with a duplicate must canonicalize equal.
    final PipelineOptions b = new PipelineOptions("en",
        PipelineOptions.Tokenizer.WHITESPACE, true, false, false, false,
        PipelineOptions.Stemmer.PORTER,
        new PipelineOptions.TermVectorSpec(PipelineOptions.TermVectorMode.FULL,
            List.of(PipelineOptions.NormalizerRung.FULL_CASE_FOLD,
                PipelineOptions.NormalizerRung.WHITESPACE,
                PipelineOptions.NormalizerRung.WHITESPACE)),
        null);

    assertThat(b).isEqualTo(a);
    assertThat(b.hashCode()).isEqualTo(a.hashCode());
  }
}
