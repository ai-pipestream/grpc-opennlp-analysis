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
import opennlp.tools.lemmatizer.DictionaryLemmatizer;
import opennlp.tools.namefind.TokenNameFinderModel;
import opennlp.tools.postag.POSModel;
import opennlp.tools.stemmer.hunspell.HunspellDictionary;

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
 * @param lemmatizer the dictionary lemmatizer, or {@code null} when
 *                   lemmatization is unavailable
 * @param loadWarnings problems encountered while loading configured models
 */
public record PipelineEnvironment(StaticEmbeddingModel embeddingModel,
                                  String embeddingsDirDescription,
                                  POSModel posModel,
                                  TokenNameFinderModel nerModel,
                                  HunspellDictionary hunspellDictionary,
                                  DictionaryLemmatizer lemmatizer,
                                  List<String> loadWarnings) {

  private static final Logger LOG = LoggerFactory.getLogger(PipelineEnvironment.class);

  /**
   * An environment with no models at all: embeddings, POS, NER, Hunspell, and
   * lemmatization are all unavailable. Useful for tests and for the pure
   * model-free deployment.
   *
   * @return the empty environment, never {@code null}
   */
  public static PipelineEnvironment empty() {
    return new PipelineEnvironment(null, null, null, null, null, null, List.of());
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

    DictionaryLemmatizer lemmatizer = null;
    if (config.lemmatizerDictPath() != null) {
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

    return new PipelineEnvironment(embeddingModel, embeddingsDir, posModel, nerModel,
        hunspellDictionary, lemmatizer, List.copyOf(warnings));
  }
}
