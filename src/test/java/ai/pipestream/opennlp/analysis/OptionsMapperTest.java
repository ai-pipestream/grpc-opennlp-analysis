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

import java.util.Arrays;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import ai.pipestream.opennlp.analysis.pipeline.PipelineOptions;
import ai.pipestream.opennlp.analysis.v1.AnalysisOptions;
import ai.pipestream.opennlp.analysis.v1.TermVectorOptions;

/**
 * Proto-to-pipeline mapping coverage: every wire enum value lands on the
 * same-named pipeline enum value, and UNSPECIFIED values resolve to the
 * documented defaults.
 */
class OptionsMapperTest {

  @ParameterizedTest(name = "{0} maps to the same-named pipeline rung")
  @EnumSource(value = TermVectorOptions.NormalizerRung.class,
      names = {"NORMALIZER_RUNG_UNSPECIFIED", "UNRECOGNIZED"},
      mode = EnumSource.Mode.EXCLUDE)
  void everyProtoRungMapsToAPipelineRung(TermVectorOptions.NormalizerRung rung) {
    final PipelineOptions options = OptionsMapper.fromProto(
        AnalysisOptions.newBuilder()
            .setTermVectors(TermVectorOptions.newBuilder()
                .setEnabled(true)
                .addRungs(rung))
            .build());

    assertThat(options.termVectors().rungs())
        .containsExactly(PipelineOptions.NormalizerRung.valueOf(
            rung.name().substring("NORMALIZER_RUNG_".length())));
  }

  @Test
  void unspecifiedRungsAreIgnored() {
    final PipelineOptions options = OptionsMapper.fromProto(
        AnalysisOptions.newBuilder()
            .setTermVectors(TermVectorOptions.newBuilder()
                .setEnabled(true)
                .addRungs(TermVectorOptions.NormalizerRung.NORMALIZER_RUNG_UNSPECIFIED)
                .addRungs(TermVectorOptions.NormalizerRung.NORMALIZER_RUNG_DASHES))
            .build());

    assertThat(options.termVectors().rungs())
        .containsExactly(PipelineOptions.NormalizerRung.DASHES);
  }

  @Test
  void protoAndPipelineRungEnumsStayInLockstep() {
    // The mapping is name-based; this test fails loudly when one enum gains a
    // value the other lacks, instead of silently dropping a rung at runtime.
    assertThat(Arrays.stream(TermVectorOptions.NormalizerRung.values())
        .filter(r -> r != TermVectorOptions.NormalizerRung.NORMALIZER_RUNG_UNSPECIFIED
            && r != TermVectorOptions.NormalizerRung.UNRECOGNIZED)
        .map(r -> r.name().substring("NORMALIZER_RUNG_".length())))
        .containsExactlyInAnyOrderElementsOf(
            Arrays.stream(PipelineOptions.NormalizerRung.values())
                .map(Enum::name).toList());
  }
}
