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

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import ai.pipestream.opennlp.analysis.config.ServiceConfig;
import ai.pipestream.opennlp.analysis.pipeline.PipelineEnvironment;
import ai.pipestream.opennlp.analysis.v1.AnalysisOptions;
import ai.pipestream.opennlp.analysis.v1.AnalysisServiceGrpc;
import ai.pipestream.opennlp.analysis.v1.AnalyzeRequest;
import ai.pipestream.opennlp.analysis.v1.AnalyzeResponse;
import ai.pipestream.opennlp.analysis.v1.GetCapabilitiesRequest;
import io.grpc.ManagedChannel;
import io.grpc.Server;
import io.grpc.inprocess.InProcessChannelBuilder;
import io.grpc.inprocess.InProcessServerBuilder;

/**
 * Noise scoring and text-artifact flagging over the wire: both are model-free
 * and always available, findings carry original-text spans, and the layers
 * stay empty unless requested.
 */
class NoiseAndArtifactsTest {

  private Server server;
  private ManagedChannel channel;
  private AnalysisServiceGrpc.AnalysisServiceBlockingStub stub;

  @BeforeEach
  void setUp() throws Exception {
    final String name = InProcessServerBuilder.generateName();
    server = InProcessServerBuilder.forName(name).directExecutor()
        .addService(new AnalysisServiceImpl(PipelineEnvironment.empty(),
            new ServiceConfig(0, 1024 * 1024, null, null, null, null, null, null)))
        .build().start();
    channel = InProcessChannelBuilder.forName(name).directExecutor().build();
    stub = AnalysisServiceGrpc.newBlockingStub(channel);
  }

  @AfterEach
  void tearDown() {
    channel.shutdownNow();
    server.shutdownNow();
  }

  private AnalyzeResponse analyze(String text, AnalysisOptions options) {
    return stub.analyze(AnalyzeRequest.newBuilder()
        .setText(text)
        .setOptions(options)
        .build());
  }

  @Test
  void noiseOnCleanTextIsEmpty() {
    final AnalyzeResponse response = analyze(
        "The quick brown fox jumps over the lazy dog.",
        AnalysisOptions.newBuilder().setNoise(true).build());

    assertThat(response.getNoiseList()).isEmpty();
    assertThat(response.getWarningsList()).isEmpty();
  }

  @Test
  void noiseFlagsAVowellessConsonantRunAsGibberish() {
    final AnalyzeResponse response = analyze("the bcdfghjklmnpq end",
        AnalysisOptions.newBuilder().setNoise(true).build());

    assertThat(response.getNoiseList()).singleElement().satisfies(finding -> {
      assertThat(finding.getSeverity()).isEqualTo("gibberish");
      assertThat(finding.getScore()).isGreaterThan(0.0).isLessThanOrEqualTo(1.0);
      // Original-text span: the token starts after "the ".
      assertThat(finding.getSpan().getStart()).isEqualTo(4);
      assertThat(finding.getSpan().getEnd()).isEqualTo(17);
    });
  }

  @Test
  void noiseFlagsBase64ShapedStretchesAsBinaryish() {
    final AnalyzeResponse response = analyze(
        "see aGVsbG8gd29ybGQxMjM0NTY3ODkwMTIz end",
        AnalysisOptions.newBuilder().setNoise(true).build());

    assertThat(response.getNoiseList()).singleElement().satisfies(finding -> {
      assertThat(finding.getSeverity()).isEqualTo("binaryish");
      assertThat(finding.getSpan().getStart()).isEqualTo(4);
      assertThat(finding.getSpan().getEnd()).isEqualTo(36);
    });
  }

  @Test
  void artifactsFlagReplacementCharactersAndMojibake() {
    final AnalyzeResponse replacement = analyze("bad �text here",
        AnalysisOptions.newBuilder().setArtifacts(true).build());
    assertThat(replacement.getArtifactsList()).singleElement().satisfies(artifact -> {
      assertThat(artifact.getType()).isEqualTo("replacement");
      assertThat(artifact.getSpan().getStart()).isEqualTo(4);
      assertThat(artifact.getSpan().getEnd()).isEqualTo(5);
    });

    final AnalyzeResponse mojibake = analyze("the cafÃ© was",
        AnalysisOptions.newBuilder().setArtifacts(true).build());
    assertThat(mojibake.getArtifactsList()).singleElement().satisfies(artifact -> {
      assertThat(artifact.getType()).isEqualTo("mojibake");
      assertThat(artifact.getSpan().getStart()).isEqualTo(7);
      assertThat(artifact.getSpan().getEnd()).isEqualTo(9);
    });
  }

  @Test
  void artifactsOnCleanTextIsEmpty() {
    final AnalyzeResponse response = analyze("The quick brown fox.",
        AnalysisOptions.newBuilder().setArtifacts(true).build());

    assertThat(response.getArtifactsList()).isEmpty();
    assertThat(response.getWarningsList()).isEmpty();
  }

  @Test
  void layersStayEmptyUnlessRequested() {
    final AnalyzeResponse response = analyze("bad �text bcdfghjklmnpq",
        AnalysisOptions.newBuilder().build());

    assertThat(response.getNoiseList()).isEmpty();
    assertThat(response.getArtifactsList()).isEmpty();
  }

  @Test
  void capabilitiesCoverNoiseAndArtifacts() {
    // Both are model-free: nothing to report as unavailable, and no warning
    // may name them.
    final var capabilities = stub.getCapabilities(GetCapabilitiesRequest.getDefaultInstance());
    assertThat(capabilities.getWarningsList())
        .noneSatisfy(w -> assertThat(w).containsIgnoringCase("noise"))
        .noneSatisfy(w -> assertThat(w).containsIgnoringCase("artifact"));
  }
}
