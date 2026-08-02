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

  @ParameterizedTest(name = "{0} maps to the same-named pipeline step")
  @EnumSource(value = TermVectorOptions.NormalizerStep.class,
      names = {"NORMALIZER_STEP_UNSPECIFIED", "UNRECOGNIZED"},
      mode = EnumSource.Mode.EXCLUDE)
  void everyProtoStepMapsToAPipelineStep(TermVectorOptions.NormalizerStep step) {
    final PipelineOptions options = OptionsMapper.fromProto(
        AnalysisOptions.newBuilder()
            .setTermVectors(TermVectorOptions.newBuilder()
                .setEnabled(true)
                .addSteps(step))
            .build());

    assertThat(options.termVectors().steps())
        .containsExactly(PipelineOptions.NormalizerStep.valueOf(
            step.name().substring("NORMALIZER_STEP_".length())));
  }

  @Test
  void unspecifiedStepsAreIgnored() {
    final PipelineOptions options = OptionsMapper.fromProto(
        AnalysisOptions.newBuilder()
            .setTermVectors(TermVectorOptions.newBuilder()
                .setEnabled(true)
                .addSteps(TermVectorOptions.NormalizerStep.NORMALIZER_STEP_UNSPECIFIED)
                .addSteps(TermVectorOptions.NormalizerStep.NORMALIZER_STEP_DASHES))
            .build());

    assertThat(options.termVectors().steps())
        .containsExactly(PipelineOptions.NormalizerStep.DASHES);
  }

  @Test
  void protoAndPipelineStepEnumsStayInLockstep() {
    // The mapping is name-based; this test fails loudly when one enum gains a
    // value the other lacks, instead of silently dropping a step at runtime.
    assertThat(Arrays.stream(TermVectorOptions.NormalizerStep.values())
        .filter(r -> r != TermVectorOptions.NormalizerStep.NORMALIZER_STEP_UNSPECIFIED
            && r != TermVectorOptions.NormalizerStep.UNRECOGNIZED)
        .map(r -> r.name().substring("NORMALIZER_STEP_".length())))
        .containsExactlyInAnyOrderElementsOf(
            Arrays.stream(PipelineOptions.NormalizerStep.values())
                .map(Enum::name).toList());
  }
}
