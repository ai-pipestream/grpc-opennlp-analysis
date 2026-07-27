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

import java.util.Arrays;
import java.util.List;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import opennlp.tools.document.Annotation;
import opennlp.tools.document.Document;
import opennlp.tools.document.Layers;
import opennlp.tools.stemmer.StemmerAnnotator;

/**
 * Every algorithmic stemmer in the enum is constructed and smoke-stemmed:
 * one stem per token, never blank. Spot checks pin the exact stems where the
 * algorithm makes them deterministic.
 */
class StemmerCoverageTest {

  private static final String TEXT = "running dogs quickly";

  private static AnalysisPipeline pipeline(PipelineOptions.Stemmer stemmer) {
    return AnalysisPipeline.create(
        new PipelineOptions("en", PipelineOptions.Tokenizer.WHITESPACE, false,
            false, false, false, stemmer, null, null),
        PipelineEnvironment.empty());
  }

  @ParameterizedTest(name = "{0} stems one value per token")
  @EnumSource(value = PipelineOptions.Stemmer.class, names = {"NONE", "HUNSPELL"},
      mode = EnumSource.Mode.EXCLUDE)
  void everyAlgorithmicStemmerStemsEveryToken(PipelineOptions.Stemmer stemmer) {
    final Document document = pipeline(stemmer).analyze(TEXT);

    final List<Annotation<String>> stems = document.get(StemmerAnnotator.STEMS);
    assertThat(stems).hasSameSizeAs(document.get(Layers.TOKENS));
    assertThat(stems).allSatisfy(s -> assertThat(s.value()).isNotBlank());
  }

  @Test
  void snowballEnglishStemsRunningToRun() {
    assertThat(pipeline(PipelineOptions.Stemmer.SNOWBALL_ENGLISH)
        .analyze(TEXT).get(StemmerAnnotator.STEMS))
        .extracting(Annotation::value)
        .containsExactly("run", "dog", "quick");
  }

  @Test
  void snowballPorterProgramStemsLikeClassicPorter() {
    assertThat(pipeline(PipelineOptions.Stemmer.SNOWBALL_PORTER)
        .analyze(TEXT).get(StemmerAnnotator.STEMS))
        .extracting(Annotation::value)
        .containsExactly("run", "dog", "quickli");
  }

  @Test
  void lightGermanStemsHauserToHaus() {
    assertThat(pipeline(PipelineOptions.Stemmer.LIGHT_GERMAN)
        .analyze("Häuser Hause").get(StemmerAnnotator.STEMS))
        .extracting(Annotation::value)
        .containsExactly("Haus", "Haus");
  }

  @Test
  void norwegianVarietiesBothStem() {
    for (PipelineOptions.Stemmer stemmer : Arrays.asList(
        PipelineOptions.Stemmer.LIGHT_NORWEGIAN_BOKMAAL,
        PipelineOptions.Stemmer.LIGHT_NORWEGIAN_NYNORSK,
        PipelineOptions.Stemmer.MINIMAL_NORWEGIAN_BOKMAAL,
        PipelineOptions.Stemmer.MINIMAL_NORWEGIAN_NYNORSK)) {
      assertThat(pipeline(stemmer).analyze("husene").get(StemmerAnnotator.STEMS))
          .extracting(Annotation::value)
          .allSatisfy(s -> assertThat(s).isNotBlank());
    }
  }

  @Test
  void hunspellWithoutDictionaryWarnsAndProducesNoStems() {
    final AnalysisPipeline pipeline = pipeline(PipelineOptions.Stemmer.HUNSPELL);

    assertThat(pipeline.warnings()).anySatisfy(w -> assertThat(w).contains("Hunspell"));
    assertThat(pipeline.analyze(TEXT).get(StemmerAnnotator.STEMS)).isEmpty();
    assertThat(pipeline.analyze(TEXT).get(Layers.TOKENS)).hasSize(3);
  }
}
