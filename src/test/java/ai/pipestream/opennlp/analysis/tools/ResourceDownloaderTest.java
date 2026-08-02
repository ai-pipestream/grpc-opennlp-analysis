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

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.nio.file.Path;

import org.junit.jupiter.api.Test;

/**
 * The downloader's offline surface: the catalog's integrity and the CLI's
 * argument handling. The network path itself is exercised by operators; the
 * upstream ResourceInstaller carries its own hardening tests.
 */
class ResourceDownloaderTest {

  @Test
  void hunspellEntryPointsAtTwoHttpsFilesAndBothEnvVars() {
    final var entry = ResourceCatalog.find("hunspell-en-us").orElseThrow();

    assertThat(entry.sources()).hasSize(2);
    assertThat(entry.sources())
        .allSatisfy(uri -> assertThat(uri.getScheme()).isEqualTo("https"));
    assertThat(entry.sources().toString()).contains("index.aff");
    assertThat(entry.sources().toString()).contains("index.dic");
    assertThat(entry.envVars()).containsExactlyInAnyOrder(
        "OPENNLP_HUNSPELL_AFF=index.aff", "OPENNLP_HUNSPELL_DIC=index.dic");
  }

  @Test
  void everyCatalogEntryIsWellFormed() {
    assertThat(ResourceCatalog.names()).isNotEmpty();
    for (String name : ResourceCatalog.names()) {
      final var entry = ResourceCatalog.find(name).orElseThrow();
      assertThat(entry.sources()).isNotEmpty();
      assertThat(entry.sources())
          .allSatisfy(uri -> assertThat(uri.getScheme()).isEqualTo("https"));
      assertThat(entry.envVars()).allSatisfy(pair -> {
        assertThat(pair).contains("=");
        assertThat(pair.substring(0, pair.indexOf('='))).startsWith("OPENNLP_");
      });
    }
  }

  @Test
  void envGuidanceResolvesFilesAgainstTheInstallDir() {
    final var entry = ResourceCatalog.find("hunspell-en-us").orElseThrow();

    assertThat(ResourceCatalog.envGuidance(entry, Path.of("/models/hun")))
        .containsExactlyInAnyOrder(
            "OPENNLP_HUNSPELL_AFF=/models/hun/index.aff",
            "OPENNLP_HUNSPELL_DIC=/models/hun/index.dic");
  }

  @Test
  void modelEntriesNameTheirServerSetting() {
    assertThat(ResourceCatalog.find("pos-en-ud-ewt").orElseThrow().envVars())
        .singleElement()
        .satisfies(pair -> assertThat(pair)
            .startsWith("OPENNLP_POS_MODEL=").contains(".bin"));
    assertThat(ResourceCatalog.find("ner-en-person").orElseThrow().envVars())
        .singleElement()
        .satisfies(pair -> assertThat(pair)
            .startsWith("OPENNLP_NER_MODEL=").contains("en-ner-person.bin"));
    assertThat(ResourceCatalog.find("ner-en-location").orElseThrow().envVars())
        .singleElement()
        .satisfies(pair -> assertThat(pair).startsWith("OPENNLP_NER_MODEL="));
    assertThat(ResourceCatalog.find("wordnet-en").orElseThrow().envVars())
        .singleElement()
        .satisfies(pair -> assertThat(pair).isEqualTo("OPENNLP_WORDNET_DIR=dict"));
  }

  @Test
  void morfologikEntryNamesItsServerSettingAndInJarPath() {
    final var entry = ResourceCatalog.find("morfologik-en").orElseThrow();

    // The source is the LanguageTool jar; the installer unpacks zip-magic
    // downloads with their structure, so the setting points at the in-jar
    // path of the dictionary (its .info sibling lands alongside).
    assertThat(entry.sources()).singleElement()
        .satisfies(uri -> assertThat(uri.toString())
            .contains("org/languagetool/language-en/5.9"));
    assertThat(entry.envVars()).singleElement()
        .satisfies(pair -> assertThat(pair).isEqualTo(
            "OPENNLP_MORFOLOGIK_DICT=org/languagetool/resource/en/english.dict"));
    assertThat(ResourceCatalog.envGuidance(entry, Path.of("/models/morfo")))
        .containsExactly("OPENNLP_MORFOLOGIK_DICT="
            + "/models/morfo/org/languagetool/resource/en/english.dict");
  }

  @Test
  void blankCatalogNamesAreRejected() {
    assertThatThrownBy(() -> new ResourceCatalog.Entry(" ", "d",
        java.util.List.of(java.net.URI.create("https://example.org/x")), null))
        .isInstanceOf(IllegalArgumentException.class);
  }

  @Test
  void unknownNamesFailLoudly() {
    assertThat(ResourceCatalog.find("no-such-thing")).isEmpty();
    assertThat(ResourceDownloader.run(new String[] {"install", "no-such-thing"}))
        .isEqualTo(2);
  }

  @Test
  void listExitsCleanly() {
    assertThat(ResourceDownloader.run(new String[0])).isZero();
    assertThat(ResourceDownloader.run(new String[] {"list"})).isZero();
  }

  @Test
  void urlInstallWithoutTargetDirIsAUsageError() {
    assertThat(ResourceDownloader.run(
        new String[] {"install", "https://example.org/x.tar.gz"}))
        .isEqualTo(2);
  }

  @Test
  void garbageArgsAreAUsageError() {
    assertThat(ResourceDownloader.run(new String[] {"frobnicate"})).isEqualTo(2);
  }
}
