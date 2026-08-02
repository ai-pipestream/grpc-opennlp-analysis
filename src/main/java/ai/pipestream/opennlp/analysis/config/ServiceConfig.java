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

package ai.pipestream.opennlp.analysis.config;

import java.nio.file.Path;
import java.util.List;

/**
 * Server configuration, resolved from process arguments, environment variables,
 * and system properties.
 *
 * <p>Precedence: system property first (explicit override), then the environment
 * variable, then the default. The port is taken from the first command line
 * argument when present.</p>
 *
 * @param port the gRPC listen port
 * @param maxTextBytes maximum accepted request text size in bytes
 * @param embeddingsDir directory of the static (Model2Vec) embedding model, or
 *                      {@code null} to run with embeddings disabled
 * @param posModelPath path to a POS model file, or {@code null} when POS tagging
 *                     is not offered
 * @param nerModelPath path to a token name finder model file, or {@code null}
 *                     when NER is not offered
 * @param hunspellAffPath path to a Hunspell .aff file, or {@code null}
 * @param hunspellDicPath path to a Hunspell .dic file, or {@code null}
 * @param lemmatizerDictPath path to a lemmatizer dictionary file (token/lemma
 *                           pairs per line), or {@code null}
 * @param latticeDicDir path to a MeCab dictionary directory for the lattice
 *                      (CJK) tokenizer, or {@code null}
 * @param sentencePieceModelPath path to a SentencePiece .model file, or
 *                               {@code null}
 * @param depparseModelPath path to a feedforward dependency-parsing model
 *                          file, or {@code null}
 * @param wordnetDir path to a WordNet database directory (the WNdb
 *                   {@code data.*}/{@code index.*}/{@code *.exc} files), or
 *                   {@code null}; when set it is the lemmatizer backend,
 *                   taking precedence over {@code morfologikDictPath} and
 *                   {@code lemmatizerDictPath}
 * @param morfologikDictPath path to a morfologik morphological dictionary
 *                           (a CFSA2/FSA5 {@code .dict} automaton with its
 *                           {@code .info} metadata sibling), or {@code null};
 *                           the lemmatizer backend when no WordNet directory
 *                           is set, taking precedence over
 *                           {@code lemmatizerDictPath}
 * @param spellcheckModelPath path to a serialized SymSpell spell-check model
 *                            behind NORMALIZER_STEP_SPELLCHECK, or
 *                            {@code null}
 * @param streamWorkers number of analysis worker threads serving
 *                      AnalyzeStream calls; {@code 0} selects the
 *                      processor count
 */
public record ServiceConfig(int port, int maxTextBytes, Path embeddingsDir,
                            Path posModelPath, List<Path> nerModelPaths,
                            Path hunspellAffPath, Path hunspellDicPath,
                            Path lemmatizerDictPath, Path latticeDicDir,
                            Path sentencePieceModelPath, Path depparseModelPath,
                            Path wordnetDir, Path morfologikDictPath,
                            Path spellcheckModelPath, int streamWorkers) {

  /**
   * Normalizes the NER model list. Every other model setting says
   * "absent" with {@code null}, so callers pass {@code null} here too;
   * turning it into an empty list at the boundary keeps that idiom
   * without every reader having to null-check a collection.
   */
  public ServiceConfig {
    nerModelPaths = nerModelPaths == null ? List.of() : List.copyOf(nerModelPaths);
  }

  /** Default gRPC listen port. */
  public static final int DEFAULT_PORT = 50051;

  /** Default request text size cap: 1 MiB. */
  public static final int DEFAULT_MAX_TEXT_BYTES = 1024 * 1024;

  /** Without the tokenizer/parser models and with an auto-sized stream worker
   * pool. */
  public ServiceConfig(int port, int maxTextBytes, Path embeddingsDir,
                       Path posModelPath, Path nerModelPath,
                       Path hunspellAffPath, Path hunspellDicPath,
                       Path lemmatizerDictPath) {
    this(port, maxTextBytes, embeddingsDir, posModelPath, one(nerModelPath),
        hunspellAffPath, hunspellDicPath, lemmatizerDictPath, null, null, null, null,
        null, null, 0);
  }

  /** Auto-sized stream worker pool (the processor count). */
  public ServiceConfig(int port, int maxTextBytes, Path embeddingsDir,
                       Path posModelPath, Path nerModelPath,
                       Path hunspellAffPath, Path hunspellDicPath,
                       Path lemmatizerDictPath, Path latticeDicDir,
                       Path sentencePieceModelPath, Path depparseModelPath) {
    this(port, maxTextBytes, embeddingsDir, posModelPath, one(nerModelPath),
        hunspellAffPath, hunspellDicPath, lemmatizerDictPath, latticeDicDir,
        sentencePieceModelPath, depparseModelPath, null, null, null, 0);
  }

  /**
   * The stream worker count this configuration resolves to: the configured
   * value, or the processor count (at least two) when unset.
   *
   * @return the number of worker threads AnalyzeStream should use
   */
  public int resolvedStreamWorkers() {
    return streamWorkers > 0 ? streamWorkers
        : Math.max(2, Runtime.getRuntime().availableProcessors());
  }

  /**
   * Resolves the configuration from the process environment.
   *
   * @param args command line arguments; {@code args[0]} is the port when present
   * @return the resolved configuration, never {@code null}
   */
  public static ServiceConfig fromEnvironment(String[] args) {
    final int port;
    if (args != null && args.length > 0 && !args[0].isBlank()) {
      port = Integer.parseInt(args[0]);
    } else {
      port = Integer.parseInt(setting("PORT", "port", Integer.toString(DEFAULT_PORT)));
    }
    final int maxTextBytes = Integer.parseInt(setting(
        "OPENNLP_MAX_TEXT_BYTES", "opennlp.max.text.bytes",
        Integer.toString(DEFAULT_MAX_TEXT_BYTES)));
    return new ServiceConfig(port, maxTextBytes,
        path(setting("OPENNLP_EMBEDDINGS_DIR", "opennlp.embeddings.dir", null)),
        path(setting("OPENNLP_POS_MODEL", "opennlp.pos.model", null)),
        paths(setting("OPENNLP_NER_MODEL", "opennlp.ner.model", null)),
        path(setting("OPENNLP_HUNSPELL_AFF", "opennlp.hunspell.aff", null)),
        path(setting("OPENNLP_HUNSPELL_DIC", "opennlp.hunspell.dic", null)),
        path(setting("OPENNLP_LEMMATIZER_DICT", "opennlp.lemmatizer.dict", null)),
        path(setting("OPENNLP_LATTICE_DIC_DIR", "opennlp.lattice.dic.dir", null)),
        path(setting("OPENNLP_SENTENCEPIECE_MODEL", "opennlp.sentencepiece.model", null)),
        path(setting("OPENNLP_DEPPARSE_MODEL", "opennlp.depparse.model", null)),
        path(setting("OPENNLP_WORDNET_DIR", "opennlp.wordnet.dir", null)),
        path(setting("OPENNLP_MORFOLOGIK_DICT", "opennlp.morfologik.dict", null)),
        path(setting("OPENNLP_SPELLCHECK_MODEL", "opennlp.spellcheck.model", null)),
        Integer.parseInt(setting(
            "OPENNLP_STREAM_WORKERS", "opennlp.stream.workers", "0")));
  }

  private static String setting(String envVar, String property, String fallback) {
    final String fromProperty = System.getProperty(property);
    if (fromProperty != null && !fromProperty.isBlank()) {
      return fromProperty;
    }
    final String fromEnv = System.getenv(envVar);
    if (fromEnv != null && !fromEnv.isBlank()) {
      return fromEnv;
    }
    return fallback;
  }

  private static Path path(String value) {
    return value == null ? null : Path.of(value);
  }

  /** A single optional path as the list form the record now holds. */
  private static List<Path> one(Path value) {
    return value == null ? List.of() : List.of(value);
  }

  /**
   * Several model paths from one setting. The stock English NER models are
   * one file per entity type, so naming only one of them silently limits
   * what the service can find: a person-only deployment reports no
   * locations and no organizations while looking perfectly healthy.
   * Separate with the platform path separator or a comma. A blank setting
   * is no models rather than one empty path.
   */
  private static List<Path> paths(String value) {
    if (value == null || value.isBlank()) {
      return List.of();
    }
    return java.util.Arrays.stream(value.split("[" + java.io.File.pathSeparator + ",]"))
        .map(String::trim)
        .filter(s -> !s.isEmpty())
        .map(Path::of)
        .toList();
  }
}
