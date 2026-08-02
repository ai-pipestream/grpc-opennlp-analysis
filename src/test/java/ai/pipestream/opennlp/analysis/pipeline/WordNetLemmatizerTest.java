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

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.nio.file.Path;
import java.util.List;

import org.junit.jupiter.api.Test;

import opennlp.wordnet.MorphyExceptions;
import opennlp.wordnet.MorphyLemmatizer;
import opennlp.wordnet.WndbReader;

/**
 * The WordNet lemmatizer adapter over the mini WordNet fixture (nouns
 * "dog"/"man"/"run", verbs "go"/"run"/"walk", exception lists
 * "men -> man", "running -> run", …): UD tag mapping, the tagless all-POS
 * fallback, and the respected-tag rule.
 */
class WordNetLemmatizerTest {

  private static WordNetLemmatizer lemmatizer() throws Exception {
    final Path dir = Path.of(WordNetLemmatizerTest.class
        .getResource("/opennlp/wordnet/mini-wndb").toURI());
    return new WordNetLemmatizer(
        new MorphyLemmatizer(WndbReader.read(dir), MorphyExceptions.load(dir)));
  }

  @Test
  void udTagsLemmatizeLikeTheirPennCounterparts() throws Exception {
    final WordNetLemmatizer lemmatizer = lemmatizer();

    // Rule-derived noun plural, validated against the lexicon.
    assertThat(lemmatizer.lemmatize(new String[] {"dogs"}, new String[] {"NOUN"}))
        .containsExactly("dog");
    // Exception-list verb form.
    assertThat(lemmatizer.lemmatize(new String[] {"running"}, new String[] {"VERB"}))
        .containsExactly("run");
    // PROPN is the UD tag Morphy's Penn-prefix mapping misses; the adapter
    // rewrites it so proper nouns lemmatize as nouns.
    assertThat(lemmatizer.lemmatize(new String[] {"dogs"}, new String[] {"PROPN"}))
        .containsExactly("dog");
    // Morphy folds case before lookup.
    assertThat(lemmatizer.lemmatize(new String[] {"Dogs"}, new String[] {"NOUN"}))
        .containsExactly("dog");
  }

  @Test
  void taglessTokensFallBackAcrossAllPartsOfSpeech() throws Exception {
    final WordNetLemmatizer lemmatizer = lemmatizer();

    // Noun exception list, found on the first fallback step.
    assertThat(lemmatizer.lemmatize(new String[] {"men"}, new String[] {""}))
        .containsExactly("man");
    // Not a noun in the fixture, so the fallback continues to the verb
    // exception list.
    assertThat(lemmatizer.lemmatize(new String[] {"running"}, new String[] {""}))
        .containsExactly("run");
    // A lexicon word lemmatizes to itself.
    assertThat(lemmatizer.lemmatize(new String[] {"box"}, new String[] {""}))
        .containsExactly("box");
    // Unknown everywhere: the OpenNLP unknown marker.
    assertThat(lemmatizer.lemmatize(new String[] {"zebra"}, new String[] {""}))
        .containsExactly("O");
  }

  @Test
  void aMappableTagIsRespectedWithoutFallback() throws Exception {
    final WordNetLemmatizer lemmatizer = lemmatizer();

    // "running" is a verb exception but not a noun in the fixture; a real
    // NOUN tag is respected, so the result is unknown rather than the verb
    // lemma.
    assertThat(lemmatizer.lemmatize(new String[] {"running"}, new String[] {"NOUN"}))
        .containsExactly("O");
  }

  @Test
  void invalidArgumentsAreRejected() throws Exception {
    final WordNetLemmatizer lemmatizer = lemmatizer();

    assertThatThrownBy(() -> lemmatizer.lemmatize(null, new String[] {""}))
        .isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(() -> lemmatizer.lemmatize(new String[] {"dogs"}, null))
        .isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(() -> lemmatizer.lemmatize(
        new String[] {"dogs", "men"}, new String[] {"NOUN"}))
        .isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(() -> lemmatizer.lemmatize(
        new String[] {"dogs"}, new String[] {null}))
        .isInstanceOf(IllegalArgumentException.class);
    assertThat(lemmatizer.lemmatize(List.of("dogs"), List.of("NOUN")))
        .containsExactly(List.of("dog"));
  }
}
