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

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ai.pipestream.opennlp.analysis.config.ServiceConfig;
import opennlp.embeddings.StaticEmbeddingModel;
import opennlp.subword.sentencepiece.SentencePieceTokenizer;
import opennlp.tools.depparse.FeedforwardDependencyModel;
import opennlp.tools.lemmatizer.DictionaryLemmatizer;
import opennlp.tools.lemmatizer.Lemmatizer;
import opennlp.tools.namefind.TokenNameFinderModel;
import opennlp.tools.postag.POSModel;
import opennlp.tools.stemmer.hunspell.HunspellDictionary;
import opennlp.tools.tokenize.lattice.MecabDictionary;
import opennlp.spellcheck.dictionary.SymSpellModel;
import opennlp.spellcheck.dictionary.SymSpellModels;
import opennlp.wordnet.MorphyExceptions;
import opennlp.wordnet.MorphyLemmatizer;
import opennlp.wordnet.WndbReader;

/**
 * Server-level shared resources that pipelines draw on: the static embedding
 * model, the optional POS/NER models, the optional Hunspell dictionary, and
 * the optional lemmatizer dictionary. Loaded once at startup and shared by
 * every cached pipeline. All fields may be {@code null}; a {@code null} field
 * means the corresponding feature is unavailable and requests for it resolve
 * to a warning, never to a runtime model download.
 *
 * @param embeddingModel the static embedding model, or {@code null} when
 *                       embeddings are disabled
 * @param embeddingsDirDescription description of the directory the embedding
 *                                 model was loaded from, or {@code null}
 * @param posModel the POS model, or {@code null} when POS is unavailable
 * @param nerModel the NER model, or {@code null} when NER is unavailable
 * @param hunspellDictionary the Hunspell dictionary, or {@code null} when
 *                           STEMMER_HUNSPELL is unavailable
 * @param lemmatizer the lemmatizer backend (WordNet Morphy when
 *                   {@code OPENNLP_WORDNET_DIR} is set, otherwise the
 *                   dictionary lemmatizer), or {@code null} when lemmatization
 *                   is unavailable
 * @param mecabDictionary the MeCab dictionary for the lattice (CJK)
 *                        tokenizer, or {@code null} when TOKENIZER_LATTICE is
 *                        unavailable
 * @param sentencePieceTokenizer the SentencePiece model for the subword
 *                               tokenizer, or {@code null} when
 *                               TOKENIZER_SENTENCEPIECE is unavailable
 * @param depparseModel the feedforward dependency-parsing model, or
 *                      {@code null} when dependency parsing is unavailable
 * @param spellcheckModel the SymSpell spell-check model behind
 *                        NORMALIZER_STEP_SPELLCHECK, or {@code null} when
 *                        spell correction is unavailable
 * @param loadWarnings problems encountered while loading configured models
 */
public record PipelineEnvironment(StaticEmbeddingModel embeddingModel,
                                  String embeddingsDirDescription,
                                  POSModel posModel,
                                  TokenNameFinderModel nerModel,
                                  HunspellDictionary hunspellDictionary,
                                  Lemmatizer lemmatizer,
                                  MecabDictionary mecabDictionary,
                                  SentencePieceTokenizer sentencePieceTokenizer,
                                  FeedforwardDependencyModel depparseModel,
                                  SymSpellModel spellcheckModel,
                                  List<String> loadWarnings) {

  private static final Logger LOG = LoggerFactory.getLogger(PipelineEnvironment.class);

  /**
   * Without the tokenizer/parser models: lattice, SentencePiece, and
   * dependency parsing are unavailable.
   */
  public PipelineEnvironment(StaticEmbeddingModel embeddingModel,
                             String embeddingsDirDescription,
                             POSModel posModel,
                             TokenNameFinderModel nerModel,
                             HunspellDictionary hunspellDictionary,
                             Lemmatizer lemmatizer,
                             List<String> loadWarnings) {
    this(embeddingModel, embeddingsDirDescription, posModel, nerModel,
        hunspellDictionary, lemmatizer, null, null, null, null, loadWarnings);
  }

  /**
   * An environment with no models at all: embeddings, POS, NER, Hunspell, and
   * lemmatization are all unavailable. Useful for tests and for the pure
   * model-free deployment.
   *
   * @return the empty environment, never {@code null}
   */
  public static PipelineEnvironment empty() {
    return new PipelineEnvironment(null, null, null, null, null, null, null, null, null,
        null, List.of());
  }

  /**
   * Loads every model the configuration points at. A configured model that
   * fails to load is reported in {@link #loadWarnings()} and logged; the
   * server still starts with the feature disabled.
   *
   * @param config the server configuration
   * @return the loaded environment, never {@code null}
   */
  public static PipelineEnvironment load(ServiceConfig config) {
    final List<String> warnings = new ArrayList<>();

    StaticEmbeddingModel embeddingModel = null;
    String embeddingsDir = null;
    if (config.embeddingsDir() != null) {
      try {
        embeddingModel = StaticEmbeddingModel.load(config.embeddingsDir());
        embeddingsDir = config.embeddingsDir().toString();
        LOG.info("Loaded embedding model from {} (dimension {})",
            embeddingsDir, embeddingModel.dimension());
      } catch (IOException | RuntimeException e) {
        warnings.add("OPENNLP_EMBEDDINGS_DIR is set to " + config.embeddingsDir()
            + " but the model could not be loaded: " + e.getMessage()
            + "; embedding requests return a warning and no vectors");
        LOG.error("Could not load embedding model from {}", config.embeddingsDir(), e);
      }
    } else {
      warnings.add("OPENNLP_EMBEDDINGS_DIR is not set; embedding requests return "
          + "a warning and no vectors");
    }

    POSModel posModel = null;
    if (config.posModelPath() != null) {
      try {
        posModel = new POSModel(config.posModelPath());
        LOG.info("Loaded POS model from {}", config.posModelPath());
      } catch (IOException | RuntimeException e) {
        warnings.add("OPENNLP_POS_MODEL is set to " + config.posModelPath()
            + " but the model could not be loaded: " + e.getMessage()
            + "; pos_tags requests return a warning and no tags");
        LOG.error("Could not load POS model from {}", config.posModelPath(), e);
      }
    }

    TokenNameFinderModel nerModel = null;
    if (config.nerModelPath() != null) {
      try {
        nerModel = new TokenNameFinderModel(config.nerModelPath());
        LOG.info("Loaded NER model from {}", config.nerModelPath());
      } catch (IOException | RuntimeException e) {
        warnings.add("OPENNLP_NER_MODEL is set to " + config.nerModelPath()
            + " but the model could not be loaded: " + e.getMessage()
            + "; ner requests return a warning and no entities");
        LOG.error("Could not load NER model from {}", config.nerModelPath(), e);
      }
    }

    HunspellDictionary hunspellDictionary = null;
    if (config.hunspellAffPath() != null && config.hunspellDicPath() != null) {
      try {
        hunspellDictionary =
            HunspellDictionary.load(config.hunspellAffPath(), config.hunspellDicPath());
        LOG.info("Loaded Hunspell dictionary from {} + {}",
            config.hunspellAffPath(), config.hunspellDicPath());
      } catch (IOException | RuntimeException e) {
        warnings.add("Hunspell is configured (" + config.hunspellAffPath() + ", "
            + config.hunspellDicPath() + ") but the dictionary could not be loaded: "
            + e.getMessage() + "; STEMMER_HUNSPELL requests return a warning and no stems");
        LOG.error("Could not load Hunspell dictionary", e);
      }
    } else if (config.hunspellAffPath() != null || config.hunspellDicPath() != null) {
      warnings.add("Hunspell needs both OPENNLP_HUNSPELL_AFF and OPENNLP_HUNSPELL_DIC; "
          + "only one is set, so STEMMER_HUNSPELL requests return a warning and no stems");
    }

    Lemmatizer lemmatizer = null;
    if (config.wordnetDir() != null) {
      // WordNet wins over the flat dictionary when both are configured: the
      // Morphy lemmatizer validates rule-derived candidates against the
      // lexicon, so its lemmas are real words rather than truncations.
      try {
        lemmatizer = new WordNetLemmatizer(new MorphyLemmatizer(
            WndbReader.read(config.wordnetDir()),
            MorphyExceptions.load(config.wordnetDir())));
        LOG.info("Loaded WordNet lemmatizer from {}", config.wordnetDir());
        if (config.lemmatizerDictPath() != null) {
          warnings.add("both OPENNLP_WORDNET_DIR and OPENNLP_LEMMATIZER_DICT are "
              + "set; the WordNet lemmatizer serves lemmatize requests and "
              + config.lemmatizerDictPath() + " is ignored");
        }
      } catch (IOException | RuntimeException e) {
        warnings.add("OPENNLP_WORDNET_DIR is set to " + config.wordnetDir()
            + " but the WordNet database could not be loaded: " + e.getMessage()
            + "; lemmatize requests return a warning and no lemmas");
        LOG.error("Could not load WordNet database from {}", config.wordnetDir(), e);
      }
    } else if (config.lemmatizerDictPath() != null) {
      try {
        lemmatizer = new DictionaryLemmatizer(config.lemmatizerDictPath());
        LOG.info("Loaded lemmatizer dictionary from {}", config.lemmatizerDictPath());
      } catch (IOException | RuntimeException e) {
        warnings.add("OPENNLP_LEMMATIZER_DICT is set to " + config.lemmatizerDictPath()
            + " but the dictionary could not be loaded: " + e.getMessage()
            + "; lemmatize requests return a warning and no lemmas");
        LOG.error("Could not load lemmatizer dictionary from {}",
            config.lemmatizerDictPath(), e);
      }
    }

    MecabDictionary mecabDictionary = null;
    if (config.latticeDicDir() != null) {
      try {
        mecabDictionary = MecabDictionary.load(config.latticeDicDir());
        LOG.info("Loaded MeCab dictionary from {}", config.latticeDicDir());
      } catch (IOException | RuntimeException e) {
        warnings.add("OPENNLP_LATTICE_DIC_DIR is set to " + config.latticeDicDir()
            + " but the dictionary could not be loaded: " + e.getMessage()
            + "; TOKENIZER_LATTICE requests return a warning and whitespace tokens");
        LOG.error("Could not load MeCab dictionary from {}", config.latticeDicDir(), e);
      }
    }

    SentencePieceTokenizer sentencePieceTokenizer = null;
    if (config.sentencePieceModelPath() != null) {
      try {
        sentencePieceTokenizer = SentencePieceTokenizer.load(config.sentencePieceModelPath());
        LOG.info("Loaded SentencePiece model from {} ({} pieces)",
            config.sentencePieceModelPath(), sentencePieceTokenizer.vocabularySize());
      } catch (IOException | RuntimeException e) {
        warnings.add("OPENNLP_SENTENCEPIECE_MODEL is set to " + config.sentencePieceModelPath()
            + " but the model could not be loaded: " + e.getMessage()
            + "; TOKENIZER_SENTENCEPIECE requests return a warning and whitespace tokens");
        LOG.error("Could not load SentencePiece model from {}",
            config.sentencePieceModelPath(), e);
      }
    }

    FeedforwardDependencyModel depparseModel = null;
    if (config.depparseModelPath() != null) {
      try {
        depparseModel = FeedforwardDependencyModel.load(config.depparseModelPath());
        LOG.info("Loaded dependency-parsing model from {}", config.depparseModelPath());
      } catch (IOException | RuntimeException e) {
        warnings.add("OPENNLP_DEPPARSE_MODEL is set to " + config.depparseModelPath()
            + " but the model could not be loaded: " + e.getMessage()
            + "; dependency_parse requests return a warning and no arcs");
        LOG.error("Could not load dependency-parsing model from {}",
            config.depparseModelPath(), e);
      }
    }

    SymSpellModel spellcheckModel = null;
    if (config.spellcheckModelPath() != null) {
      try {
        final java.io.InputStream in = java.nio.file.Files.newInputStream(
            config.spellcheckModelPath());
        try (in) {
          spellcheckModel = SymSpellModels.deserialize(in);
        }
        LOG.info("Loaded spell-check model from {} ({} unigrams)",
            config.spellcheckModelPath(), spellcheckModel.unigrams().size());
      } catch (IOException | RuntimeException e) {
        warnings.add("OPENNLP_SPELLCHECK_MODEL is set to " + config.spellcheckModelPath()
            + " but the spell-check model could not be loaded: " + e.getMessage()
            + "; NORMALIZER_STEP_SPELLCHECK requests return a warning and no correction");
        LOG.error("Could not load spell-check model from {}",
            config.spellcheckModelPath(), e);
      }
    }

    return new PipelineEnvironment(embeddingModel, embeddingsDir, posModel, nerModel,
        hunspellDictionary, lemmatizer, mecabDictionary, sentencePieceTokenizer,
        depparseModel, spellcheckModel, List.copyOf(warnings));
  }
}
