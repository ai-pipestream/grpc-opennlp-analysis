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
import java.io.InputStream;
import java.util.Properties;

/**
 * Resolves version strings reported by GetCapabilities from classpath
 * metadata, with safe fallbacks for unpacked (IDE/test) classpaths.
 */
final class Versions {

  /** Path of the Maven metadata inside the opennlp-api jar. */
  private static final String OPENNLP_POM_PROPERTIES =
      "/META-INF/maven/ai.pipestream/opennlp-api/pom.properties";

  /** Resource written by the Gradle build with the service version. */
  private static final String SERVICE_VERSION_RESOURCE = "/service-version.properties";

  private Versions() {
  }

  /**
   * @return the version of the wrapped OpenNLP build, for example
   *         {@code "3.x-preview-SNAPSHOT"}, or {@code "unknown"}
   */
  static String opennlp() {
    return read(OPENNLP_POM_PROPERTIES, "version", "unknown");
  }

  /**
   * @return the service version baked in by the build, or {@code "dev"}
   */
  static String service() {
    return read(SERVICE_VERSION_RESOURCE, "version", "dev");
  }

  private static String read(String resource, String key, String fallback) {
    try (InputStream in = Versions.class.getResourceAsStream(resource)) {
      if (in == null) {
        return fallback;
      }
      final Properties properties = new Properties();
      properties.load(in);
      return properties.getProperty(key, fallback);
    } catch (IOException e) {
      return fallback;
    }
  }
}
