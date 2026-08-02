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

import java.io.IOException;
import java.net.URI;
import java.nio.file.Path;

import opennlp.tools.util.ResourceInstaller;

/**
 * Operator CLI for fetching the analysis resources the server reads its
 * optional features from (Hunspell dictionaries, MeCab dictionaries, model
 * files). Built on the OpenNLP preview's {@link ResourceInstaller}: fetches
 * over the network, verifies an optional SHA-256, and unpacks tar.gz/zip
 * archives or stores plain files.
 *
 * <p>Usage:</p>
 * <pre>
 *   ./gradlew downloadResource --args="list"
 *   ./gradlew downloadResource --args="install hunspell-en-us /path/to/dir"
 *   ./gradlew downloadResource --args="install https://example.org/x.tar.gz /path/to/dir [sha256-hex]"
 * </pre>
 *
 * <p>Installing a catalog entry prints the environment variables to set for
 * the server to pick the resource up. Nothing is downloaded at server
 * request time; this tool is the only fetch path, run deliberately by an
 * operator.</p>
 */
public final class ResourceDownloader {

  private ResourceDownloader() {
  }

  /**
   * CLI entry point.
   *
   * @param args see class javadoc
   */
  public static void main(String[] args) {
    System.exit(run(args));
  }

  /** Visible for testing: the exit-code path, separated from the process. */
  static int run(String[] args) {
    if (args.length == 0 || args[0].equals("list")) {
      System.out.println("Known resources:");
      for (String name : ResourceCatalog.names()) {
        ResourceCatalog.find(name).ifPresent(entry ->
            System.out.println("  " + entry.name() + " - " + entry.description()));
      }
      System.out.println("Or install any archive/file by URL: "
          + "install <uri> <dir> [sha256-hex]");
      return 0;
    }
    if (!args[0].equals("install") || args.length < 2) {
      return usageError("expected: list | install <name> [dir] | install <uri> <dir> [sha256]");
    }
    final String what = args[1];
    if (what.startsWith("https://") || what.startsWith("http://")) {
      if (args.length < 3) {
        return usageError("installing a URL needs a target directory");
      }
      final String sha256 = args.length > 3 ? args[3] : null;
      try {
        final Path dir = ResourceInstaller.install(URI.create(what), Path.of(args[2]), sha256);
        System.out.println("installed " + what + " -> " + dir.toAbsolutePath());
        return 0;
      } catch (IOException | RuntimeException e) {
        System.err.println("download failed: " + e.getMessage());
        return 1;
      }
    }
    final var entry = ResourceCatalog.find(what);
    if (entry.isEmpty()) {
      System.err.println("unknown resource: " + what);
      System.err.println("known: " + String.join(", ", ResourceCatalog.names()));
      return 2;
    }
    final Path dir = Path.of(args.length > 2 ? args[2] : what);
    try {
      for (URI source : entry.get().sources()) {
        ResourceInstaller.install(source, dir);
        System.out.println("fetched " + source);
      }
    } catch (IOException | RuntimeException e) {
      System.err.println("download failed: " + e.getMessage());
      return 1;
    }
    System.out.println("installed " + what + " into " + dir.toAbsolutePath());
    System.out.println("set for the server:");
    ResourceCatalog.envGuidance(entry.get(), dir).forEach(line -> System.out.println("  " + line));
    return 0;
  }

  private static int usageError(String message) {
    System.err.println(message);
    System.err.println("usage: list | install <name> [dir] | install <uri> <dir> [sha256-hex]");
    return 2;
  }
}
