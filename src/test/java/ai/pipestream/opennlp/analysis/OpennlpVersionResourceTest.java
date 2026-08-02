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
 * Regression guard for the fork's {@code opennlp/tools/util/opennlp.version}
 * resource: when its string fails {@code Version.parse} (as the
 * "3.x-preview-SNAPSHOT" rendition did), every BaseModel load and train
 * breaks. The fork now ships the parseable "0.0.0-preview-SNAPSHOT", which
 * deliberately equals {@link Version#DEV_VERSION} after parsing: the preview
 * claims no release lineage, and BaseModel's model-compat check stays
 * disabled, exactly as for a dev build. This test fails the moment a
 * published build regresses.
 */
class OpennlpVersionResourceTest {

  @Test
  void theClasspathOpennlpVersionParses() {
    // currentVersion() itself throws NumberFormatException when the resource
    // does not parse; reaching the assertions is most of the proof.
    final Version version = Version.currentVersion();

    assertThat(version).isEqualTo(Version.DEV_VERSION);
    assertThat(version.isSnapshot()).isTrue();
  }
}
