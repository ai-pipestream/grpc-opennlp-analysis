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

package ai.pipestream.opennlp.analysis.vocab;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashSet;
import java.util.Set;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * The surface forms an embedding model covers, read from the model directory
 * rather than from a private API: a {@code vocab.txt} with one token per
 * line, or a HuggingFace-style {@code tokenizer.json} whose {@code
 * model.vocab} maps tokens to ids (the keys are the vocabulary).
 *
 * <p>Absence is a first-class outcome: no readable vocabulary means the
 * out-of-vocabulary share is not computable, and the drift response says so
 * instead of inventing a number.</p>
 */
final class EmbeddingVocabulary {

  private static final Logger LOG = LoggerFactory.getLogger(EmbeddingVocabulary.class);

  private EmbeddingVocabulary() {
  }

  /**
   * Loads the embedding vocabulary from a model directory.
   *
   * @param embeddingsDir the embedding model directory, or {@code null}
   * @return the covered surface forms, or {@code null} when no vocabulary is
   *         readable
   */
  static Set<String> load(Path embeddingsDir) {
    if (embeddingsDir == null) {
      return null;
    }
    final Path vocabTxt = embeddingsDir.resolve("vocab.txt");
    if (Files.isRegularFile(vocabTxt)) {
      try {
        final Set<String> vocabulary = new HashSet<>(
            Files.readAllLines(vocabTxt, StandardCharsets.UTF_8));
        LOG.info("Loaded embedding vocabulary from {} ({} entries)", vocabTxt,
            vocabulary.size());
        return vocabulary;
      } catch (IOException | RuntimeException e) {
        LOG.warn("Could not read embedding vocabulary from {}: {}", vocabTxt,
            e.getMessage());
        return null;
      }
    }
    final Path tokenizerJson = embeddingsDir.resolve("tokenizer.json");
    if (Files.isRegularFile(tokenizerJson)) {
      try {
        final Set<String> vocabulary = parseModelVocab(
            Files.readString(tokenizerJson, StandardCharsets.UTF_8));
        if (vocabulary != null) {
          LOG.info("Loaded embedding vocabulary from {} ({} entries)", tokenizerJson,
              vocabulary.size());
        } else {
          LOG.warn("{} carries no readable model.vocab; embedding coverage is "
              + "not computable", tokenizerJson);
        }
        return vocabulary;
      } catch (IOException | RuntimeException e) {
        LOG.warn("Could not read embedding vocabulary from {}: {}", tokenizerJson,
            e.getMessage());
        return null;
      }
    }
    return null;
  }

  /**
   * Extracts the keys of the {@code "vocab"} member of {@code "model"} from
   * tokenizer.json. Two shapes exist in the wild: an object mapping token to
   * id (BPE-style) and an array of {@code [token, score]} pairs
   * (Unigram-style); both yield their tokens here. The scan is deliberately
   * minimal — a full JSON parser is not worth a dependency for one nested
   * member — and returns {@code null} rather than a half-parse when the
   * structure is not what it expects.
   */
  private static Set<String> parseModelVocab(String json) {
    final int modelKey = json.indexOf("\"model\"");
    if (modelKey < 0) {
      return null;
    }
    final int vocabKey = json.indexOf("\"vocab\"", modelKey);
    if (vocabKey < 0) {
      return null;
    }
    int i = vocabKey + "\"vocab\"".length();
    while (i < json.length() && (Character.isWhitespace(json.charAt(i))
        || json.charAt(i) == ':')) {
      i++;
    }
    if (i >= json.length()) {
      return null;
    }
    if (json.charAt(i) == '{') {
      return parseObjectKeys(json, i);
    }
    if (json.charAt(i) == '[') {
      return parsePairArrayFirstElements(json, i);
    }
    return null;
  }

  /** Keys of a flat JSON object starting at {@code start} (the '{'). */
  private static Set<String> parseObjectKeys(String json, int start) {
    final Set<String> keys = new HashSet<>();
    int i = start + 1;
    while (i < json.length()) {
      final char c = json.charAt(i);
      if (c == '}') {
        return keys;
      }
      if (c == '"') {
        final int end = stringEnd(json, i);
        if (end < 0) {
          return null;
        }
        int j = end + 1;
        while (j < json.length() && Character.isWhitespace(json.charAt(j))) {
          j++;
        }
        if (j < json.length() && json.charAt(j) == ':') {
          keys.add(unescape(json.substring(i + 1, end)));
          i = skipValue(json, j + 1);
          if (i < 0) {
            return null;
          }
          continue;
        }
        return null;
      }
      i++;
    }
    return null;
  }

  /** First string of each {@code [token, score]} pair, array starts at {@code start}. */
  private static Set<String> parsePairArrayFirstElements(String json, int start) {
    final Set<String> keys = new HashSet<>();
    int i = start + 1;
    while (i < json.length()) {
      final char c = json.charAt(i);
      if (c == ']') {
        return keys;
      }
      if (c == '[') {
        int j = i + 1;
        while (j < json.length() && Character.isWhitespace(json.charAt(j))) {
          j++;
        }
        if (j < json.length() && json.charAt(j) == '"') {
          final int end = stringEnd(json, j);
          if (end < 0) {
            return null;
          }
          keys.add(unescape(json.substring(j + 1, end)));
        }
        i = skipValue(json, i);
        if (i < 0) {
          return null;
        }
        continue;
      }
      i++;
    }
    return null;
  }

  /** The index of a string's closing quote, honoring backslash escapes, or -1. */
  private static int stringEnd(String json, int quoteStart) {
    int i = quoteStart + 1;
    while (i < json.length()) {
      final char c = json.charAt(i);
      if (c == '\\') {
        i += 2;
        continue;
      }
      if (c == '"') {
        return i;
      }
      i++;
    }
    return -1;
  }

  /** Skips one JSON value starting at {@code start}; returns the index just past it, or -1. */
  private static int skipValue(String json, int start) {
    int i = start;
    while (i < json.length() && Character.isWhitespace(json.charAt(i))) {
      i++;
    }
    if (i >= json.length()) {
      return -1;
    }
    final char c = json.charAt(i);
    if (c == '"') {
      final int end = stringEnd(json, i);
      return end < 0 ? -1 : end + 1;
    }
    if (c == '{' || c == '[') {
      final char open = c;
      final char close = c == '{' ? '}' : ']';
      int depth = 0;
      while (i < json.length()) {
        final char d = json.charAt(i);
        if (d == '"') {
          final int end = stringEnd(json, i);
          if (end < 0) {
            return -1;
          }
          i = end + 1;
          continue;
        }
        if (d == open) {
          depth++;
        } else if (d == close) {
          depth--;
          if (depth == 0) {
            return i + 1;
          }
        }
        i++;
      }
      return -1;
    }
    // A scalar: number, true, false, null.
    while (i < json.length() && ",}] \t\n\r".indexOf(json.charAt(i)) < 0) {
      i++;
    }
    return i;
  }

  /** Resolves the JSON string escapes a vocabulary token can realistically carry. */
  private static String unescape(String raw) {
    if (raw.indexOf('\\') < 0) {
      return raw;
    }
    final StringBuilder out = new StringBuilder(raw.length());
    for (int i = 0; i < raw.length(); i++) {
      final char c = raw.charAt(i);
      if (c == '\\' && i + 1 < raw.length()) {
        final char e = raw.charAt(++i);
        switch (e) {
          case 'n' -> out.append('\n');
          case 't' -> out.append('\t');
          case 'r' -> out.append('\r');
          case 'b' -> out.append('\b');
          case 'f' -> out.append('\f');
          case 'u' -> {
            if (i + 4 < raw.length()) {
              out.append((char) Integer.parseInt(raw.substring(i + 1, i + 5), 16));
              i += 4;
            }
          }
          default -> out.append(e);
        }
      } else {
        out.append(c);
      }
    }
    return out.toString();
  }
}
