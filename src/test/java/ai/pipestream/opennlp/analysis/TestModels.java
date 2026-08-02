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

package ai.pipestream.opennlp.analysis;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import opennlp.tools.depparse.DependencyGraph;
import opennlp.tools.depparse.DependencySample;
import opennlp.tools.depparse.FeedforwardDependencyModel;
import opennlp.tools.depparse.FeedforwardDependencyTrainer;
import opennlp.tools.namefind.NameFinderME;
import opennlp.tools.namefind.NameSample;
import opennlp.tools.namefind.TokenNameFinderFactory;
import opennlp.tools.namefind.TokenNameFinderModel;
import opennlp.tools.postag.POSModel;
import opennlp.tools.postag.POSSample;
import opennlp.tools.postag.POSTaggerFactory;
import opennlp.tools.postag.POSTaggerME;
import opennlp.tools.util.CollectionObjectStream;
import opennlp.tools.util.Span;
import opennlp.tools.util.TrainingParameters;

/**
 * Tiny in-memory trained models for the model-backed integration tests.
 * Every model here is trained from a memorized micro-corpus in
 * {@code @BeforeAll}. Each distinct sample is repeated so the maxent and
 * feedforward trainers memorize the exact sentences the tests assert on.
 */
final class TestModels {

  /** How often each distinct training sample is repeated. */
  private static final int REPETITIONS = 40;

  /** The sentence the dependency and relation tests parse, verbatim. */
  static final String RELATION_SENTENCE = "John Smith founded Acme Corp";
  static final String[] RELATION_TOKENS = {"John", "Smith", "founded", "Acme", "Corp"};
  /** The UD tags the POS model emits for {@link #RELATION_TOKENS}. */
  static final String[] RELATION_UD_TAGS = {"PROPN", "PROPN", "VERB", "PROPN", "PROPN"};
  /** The dependency heads the feedforward model is trained to reproduce. */
  static final int[] RELATION_HEADS = {1, 2, -1, 4, 2};
  /** The dependency relation labels the model is trained to reproduce. */
  static final String[] RELATION_RELS = {"compound", "nsubj", "root", "compound", "obj"};

  /** The two coref sentences, joined with a newline in the request text. */
  static final String COREF_SENTENCE_1 = "John Smith won the case";
  static final String COREF_SENTENCE_2 = "He celebrated";
  static final String COREF_TEXT = COREF_SENTENCE_1 + "\n" + COREF_SENTENCE_2;

  /** The geo sentence; the bundled gazetteer resolves "Paris". */
  static final String GEO_SENTENCE = "Paris is lovely";

  private TestModels() {
  }

  /**
   * Trains a POS tagger on Penn Treebank tags covering exactly the test
   * vocabulary; the default {@code POSTagFormat.UD} of {@link POSTaggerME}
   * maps them to UD tags (NNP to PROPN, VBD to VERB, DT to DET, PRP to PRON)
   * on output.
   *
   * @return the trained model, never {@code null}
   */
  static POSModel trainPosModel() throws IOException {
    final List<POSSample> distinct = List.of(
        new POSSample(RELATION_TOKENS, new String[] {"NNP", "NNP", "VBD", "NNP", "NNP"}),
        new POSSample(new String[] {"John", "Smith", "won", "the", "case"},
            new String[] {"NNP", "NNP", "VBD", "DT", "NN"}),
        new POSSample(new String[] {"He", "celebrated"}, new String[] {"PRP", "VBD"}),
        new POSSample(new String[] {"Paris", "is", "lovely"},
            new String[] {"NNP", "VBZ", "JJ"}),
        new POSSample(new String[] {"London", "is", "lovely"},
            new String[] {"NNP", "VBZ", "JJ"}));
    final TrainingParameters params = TrainingParameters.defaultParams();
    params.put("Iterations", "200");
    params.put("Cutoff", "1");
    return POSTaggerME.train("en",
        new CollectionObjectStream<>(repeat(distinct)), params, new POSTaggerFactory());
  }

  /**
   * Trains a single-type name finder. The "person" corpus tags "John Smith"
   * and "Acme Corp"; the "location" corpus tags "Paris" and "London".
   *
   * @param type the single entity type of the model
   * @return the trained model, never {@code null}
   */
  static TokenNameFinderModel trainNerModel(String type) throws IOException {
    final List<NameSample> distinct;
    if (type.equals("person")) {
      distinct = List.of(
          new NameSample(RELATION_TOKENS,
              new Span[] {new Span(0, 2), new Span(3, 5)}, false),
          new NameSample(new String[] {"John", "Smith", "won", "the", "case"},
              new Span[] {new Span(0, 2)}, false),
          new NameSample(new String[] {"He", "celebrated"}, new Span[0], false));
    } else if (type.equals("location")) {
      distinct = List.of(
          new NameSample(new String[] {"Paris", "is", "lovely"},
              new Span[] {new Span(0, 1)}, false),
          new NameSample(new String[] {"London", "is", "lovely"},
              new Span[] {new Span(0, 1)}, false));
    } else {
      throw new IllegalArgumentException("no test corpus for type " + type);
    }
    final TrainingParameters params = TrainingParameters.defaultParams();
    params.put("Iterations", "200");
    params.put("Cutoff", "1");
    return NameFinderME.train("en", type,
        new CollectionObjectStream<>(repeat(distinct)), params, new TokenNameFinderFactory());
  }

  /**
   * Trains the feedforward dependency parser on the relation sentence with
   * the UD tags the POS model emits for it, so the parser reproduces
   * {@link #RELATION_HEADS} and {@link #RELATION_RELS} deterministically.
   * Dropout is off and the seed fixed, following the upstream trainer test.
   *
   * @return the trained model, never {@code null}
   */
  static FeedforwardDependencyModel trainDependencyModel() throws IOException {
    final List<DependencySample> distinct = List.of(
        new DependencySample(RELATION_TOKENS, RELATION_UD_TAGS,
            DependencyGraph.of(RELATION_HEADS, RELATION_RELS)),
        new DependencySample(new String[] {"John", "Smith", "won", "the", "case"},
            new String[] {"PROPN", "PROPN", "VERB", "DET", "NOUN"},
            DependencyGraph.of(new int[] {1, 2, -1, 4, 2},
                new String[] {"compound", "nsubj", "root", "det", "obj"})));
    final FeedforwardDependencyTrainer.Settings settings =
        new FeedforwardDependencyTrainer.Settings(16, 32, 120, 32, 0.05, 0.0, 0.0, 1, 17L);
    return FeedforwardDependencyTrainer.train(
        new CollectionObjectStream<>(repeat(distinct)), settings);
  }

  /** Repeats every distinct sample so the tiny corpora are memorized. */
  private static <T> List<T> repeat(List<T> distinct) {
    final List<T> corpus = new ArrayList<>(distinct.size() * REPETITIONS);
    for (int i = 0; i < REPETITIONS; i++) {
      corpus.addAll(distinct);
    }
    return corpus;
  }
}
