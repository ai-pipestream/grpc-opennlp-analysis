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

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.Test;

import opennlp.tools.util.Version;

/**
 * The build generates a parseable rendition of the fork's broken
 * {@code opennlp/tools/util/opennlp.version} resource (its
 * "3.x-preview-SNAPSHOT" string fails {@code Version.parse}, which breaks
 * every BaseModel load and train). The app's own resources precede the
 * dependency jars, so the generated file shadows the broken one — this test
 * fails the moment that shadow stops working, for example after a fork fix
 * makes the workaround obsolete.
 */
class OpennlpVersionResourceTest {

  @Test
  void theClasspathOpennlpVersionParses() {
    // currentVersion() itself throws NumberFormatException when the resource
    // does not parse; reaching the assertions is most of the proof.
    final Version version = Version.currentVersion();

    assertThat(version.getMajor()).isEqualTo(3);
    assertThat(version.isSnapshot()).isTrue();
  }
}
