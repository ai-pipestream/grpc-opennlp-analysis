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

package ai.pipestream.opennlp.analysis.tools;

import java.net.URI;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * The named resources the downloader knows how to fetch: well-known public
 * dictionaries with their canonical locations and the server configuration
 * they map to. Every entry names its source explicitly, so installing one is
 * the operator accepting that source's license (matching the upstream
 * {@code ResourceInstaller} convention: no locations built into the library,
 * no data bundled).
 *
 * <p>Pure data and rendering; the network lives in
 * {@link ResourceDownloader}.</p>
 */
final class ResourceCatalog {

  /**
   * One named resource: the files to fetch and the environment variables the
   * server reads for it.
   *
   * @param name catalog key, for example {@code "hunspell-en-us"}
   * @param description what the resource is and where it comes from
   * @param sources the files to download, in order
   * @param envVars server settings this resource feeds, as
   *                {@code VAR_NAME=relative-file-name} pairs resolved against
   *                the install directory at print time
   */
  record Entry(String name, String description, List<URI> sources, List<String> envVars) {

    Entry {
      if (name == null || name.isBlank() || sources == null || sources.isEmpty()) {
        throw new IllegalArgumentException("catalog entries need a name and at least one source");
      }
      sources = List.copyOf(sources);
      envVars = envVars == null ? List.of() : List.copyOf(envVars);
    }
  }

  private static final Map<String, Entry> ENTRIES = Map.ofEntries(
      Map.entry("hunspell-en-us", new Entry(
          "hunspell-en-us",
          "Hunspell English (US) dictionary from the LibreOffice dictionaries "
              + "mirror (wooorm/dictionaries), for STEMMER_HUNSPELL",
          List.of(
              URI.create("https://raw.githubusercontent.com/wooorm/dictionaries/"
                  + "main/dictionaries/en/index.aff"),
              URI.create("https://raw.githubusercontent.com/wooorm/dictionaries/"
                  + "main/dictionaries/en/index.dic")),
          List.of(
              "OPENNLP_HUNSPELL_AFF=index.aff",
              "OPENNLP_HUNSPELL_DIC=index.dic"))),      Map.entry("pos-en-ud-ewt", new Entry(
          "pos-en-ud-ewt",
          "Apache OpenNLP English UD EWT part-of-speech model (Maven Central, "
              + "org.apache.opennlp:opennlp-models-pos-en), for pos_tags",
          List.of(
              URI.create("https://repo1.maven.org/maven2/org/apache/opennlp/"
                  + "opennlp-models-pos-en/1.3.0/opennlp-models-pos-en-1.3.0.jar")),
          List.of(
              "OPENNLP_POS_MODEL=opennlp-en-ud-ewt-pos-1.3-2.5.4.bin"))),      Map.entry("ner-en-person", new Entry(
          "ner-en-person",
          "Apache OpenNLP English person-name finder model (SourceForge "
              + "models-1.5), for ner. Stock English NER types cover "
              + "person/location/organization and the like; use a glossary for "
              + "anything else",
          List.of(
              URI.create("https://opennlp.sourceforge.net/models-1.5/en-ner-person.bin")),
          List.of(
              "OPENNLP_NER_MODEL=en-ner-person.bin (the setting takes a list: "
                  + "separate several models with the path separator or a comma)"))),      Map.entry("ner-en-location", new Entry(
          "ner-en-location",
          "Apache OpenNLP English location-name finder model (SourceForge "
              + "models-1.5), for ner and geo",
          List.of(
              URI.create("https://opennlp.sourceforge.net/models-1.5/"
                  + "en-ner-location.bin")),
          List.of(
              "OPENNLP_NER_MODEL=en-ner-location.bin (see ner-en-person: several "
                  + "models can be named at once)"))),      Map.entry("ner-en-organization", new Entry(
          "ner-en-organization",
          "Apache OpenNLP English organization-name finder model, for ner (SourceForge models-1.5); "
              + "one of the seven stock English types, each a separate model",
          List.of(
              URI.create("https://opennlp.sourceforge.net/models-1.5/"
                  + "en-ner-organization.bin")),
          List.of(
              "OPENNLP_NER_MODEL=en-ner-organization.bin (append to the list)"))),      Map.entry("ner-en-date", new Entry(
          "ner-en-date",
          "Apache OpenNLP English date-expression finder model, for ner (SourceForge models-1.5); "
              + "one of the seven stock English types, each a separate model",
          List.of(
              URI.create("https://opennlp.sourceforge.net/models-1.5/"
                  + "en-ner-date.bin")),
          List.of(
              "OPENNLP_NER_MODEL=en-ner-date.bin (append to the list)"))),      Map.entry("ner-en-time", new Entry(
          "ner-en-time",
          "Apache OpenNLP English time-expression finder model, for ner (SourceForge models-1.5); "
              + "one of the seven stock English types, each a separate model",
          List.of(
              URI.create("https://opennlp.sourceforge.net/models-1.5/"
                  + "en-ner-time.bin")),
          List.of(
              "OPENNLP_NER_MODEL=en-ner-time.bin (append to the list)"))),      Map.entry("ner-en-money", new Entry(
          "ner-en-money",
          "Apache OpenNLP English monetary-amount finder model, for ner (SourceForge models-1.5); "
              + "one of the seven stock English types, each a separate model",
          List.of(
              URI.create("https://opennlp.sourceforge.net/models-1.5/"
                  + "en-ner-money.bin")),
          List.of(
              "OPENNLP_NER_MODEL=en-ner-money.bin (append to the list)"))),      Map.entry("ner-en-percentage", new Entry(
          "ner-en-percentage",
          "Apache OpenNLP English percentage finder model, for ner (SourceForge models-1.5); "
              + "one of the seven stock English types, each a separate model",
          List.of(
              URI.create("https://opennlp.sourceforge.net/models-1.5/"
                  + "en-ner-percentage.bin")),
          List.of(
              "OPENNLP_NER_MODEL=en-ner-percentage.bin (append to the list)"))),      Map.entry("wordnet-en", new Entry(
          "wordnet-en",
          "Princeton WordNet 3.1 database (dict-only tarball, which unlike "
              + "WNdb-3.0.tar.gz ships the .exc exception lists Morphy needs), "
              + "for the WordNet lemmatizer backend; WordNet license "
              + "(Princeton), see "
              + "https://wordnet.princeton.edu/license-and-commercial-use",
          List.of(
              URI.create("https://wordnetcode.princeton.edu/wn3.1.dict.tar.gz")),
          List.of(
              "OPENNLP_WORDNET_DIR=dict"))),      Map.entry("morfologik-en", new Entry(
          "morfologik-en",
          "LanguageTool English morphological dictionary (CFSA2 FSA; from "
              + "org.languagetool:language-en:5.9 on Maven Central - 6.x no "
              + "longer ships it in-jar), for the morfologik lemmatizer "
              + "backend; LanguageTool is LGPL-2.1, installing is accepting "
              + "that license",
          List.of(
              URI.create("https://repo1.maven.org/maven2/org/languagetool/"
                  + "language-en/5.9/language-en-5.9.jar")),
          List.of(
              // The jar is a zip, so the installer unpacks it with its
              // structure; the .info metadata sibling ships alongside.
              "OPENNLP_MORFOLOGIK_DICT=org/languagetool/resource/en/english.dict"))));

  private ResourceCatalog() {
  }

  /** @return the catalog entry for {@code name}, or empty when unknown */
  static Optional<Entry> find(String name) {
    return Optional.ofNullable(ENTRIES.get(name));
  }

  /** @return the catalog names, sorted */
  static List<String> names() {
    return ENTRIES.keySet().stream().sorted().toList();
  }

  /**
   * Renders the environment guidance for an installed entry: one
   * {@code VAR=/absolute/path} line per setting.
   *
   * @param entry the installed entry
   * @param targetDir the directory the resource was installed into
   * @return the guidance lines, never {@code null}
   */
  static List<String> envGuidance(Entry entry, Path targetDir) {
    return entry.envVars().stream()
        .map(pair -> {
          final int eq = pair.indexOf('=');
          return pair.substring(0, eq) + "="
              + targetDir.resolve(pair.substring(eq + 1)).toAbsolutePath();
        })
        .toList();
  }
}
