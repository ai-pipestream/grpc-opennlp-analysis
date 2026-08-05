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

import static org.assertj.core.api.Assertions.assertThat;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import ai.pipestream.opennlp.analysis.v1.VocabChannel;
import ai.pipestream.opennlp.analysis.v1.VocabSnapshot;
import ai.pipestream.opennlp.analysis.v1.VocabSnapshotDescriptor;
import ai.pipestream.opennlp.analysis.vocab.VocabularyListener.TermFrequency;

/**
 * The listener's window machinery: rollover persists a snapshot and starts
 * fresh, snapshots round-trip with identical estimates, and the drift
 * metrics behave sensibly on disjoint, identical, and shifted windows.
 */
class VocabularyListenerTest {

  @TempDir
  Path vocabDir;

  private VocabularyListener listener;

  @AfterEach
  void tearDown() {
    if (listener != null) {
      listener.close();
    }
  }

  private static List<TermFrequency> terms(Object... termThenFrequency) {
    final java.util.ArrayList<TermFrequency> out = new java.util.ArrayList<>();
    for (int i = 0; i < termThenFrequency.length; i += 2) {
      out.add(new TermFrequency((String) termThenFrequency[i],
          ((Number) termThenFrequency[i + 1]).longValue()));
    }
    return out;
  }

  private static Map<VocabChannel, VocabularyListener.ChannelDrift> byChannel(
      List<VocabularyListener.ChannelDrift> drift) {
    return drift.stream().collect(Collectors.toMap(
        VocabularyListener.ChannelDrift::channel, Function.identity()));
  }

  @Test
  void rolloverPersistsASnapshotAndStartsFresh() throws Exception {
    listener = new VocabularyListener(vocabDir, 3, 16, null);
    for (int doc = 0; doc < 3; doc++) {
      listener.feed(terms("alpha", 2, "beta", 1), List.of("Alpha", "beta", "alpha"));
    }

    // The third document rolls the window over: a snapshot file exists and
    // the live window is empty again.
    try (Stream<Path> listing = Files.list(vocabDir)) {
      assertThat(listing.map(p -> p.getFileName().toString())
          .filter(n -> n.startsWith("snapshot-") && n.endsWith(".pb")))
          .hasSize(1);
    }
    assertThat(listener.windowDocuments()).isZero();
    assertThat(listener.snapshots()).hasSize(1);

    // The next document opens the next window.
    listener.feed(terms("gamma", 1), List.of("gamma"));
    assertThat(listener.windowDocuments()).isEqualTo(1);
  }

  @Test
  void snapshotRoundTripKeepsIdenticalEstimates() throws Exception {
    listener = new VocabularyListener(vocabDir, 1_000, 16, null);
    for (int doc = 0; doc < 10; doc++) {
      listener.feed(terms("court", 5, "ruling", 2, "appeal", 1),
          List.of("Court", "court", "ruling"));
    }
    final double liveCardinality = listener.liveStats().stream()
        .filter(s -> s.getChannel() == VocabChannel.VOCAB_CHANNEL_TERMS)
        .findFirst().orElseThrow().getCardinalityEstimate();

    final VocabSnapshotDescriptor descriptor = listener.snapshotNow();
    assertThat(descriptor).isNotNull();

    // Round-trip through the persisted bytes: same cardinality, same heavy
    // hitters, same count-min point estimates.
    final Path file = vocabDir.resolve(descriptor.getName());
    final VocabSnapshot snapshot = VocabSnapshot.parseFrom(Files.readAllBytes(file));
    final WindowState restored = WindowState.fromProto(snapshot.getChannelsList().stream()
        .filter(c -> c.getChannel() == VocabChannel.VOCAB_CHANNEL_TERMS)
        .findFirst().orElseThrow());
    assertThat(restored.cardinalityEstimate()).isEqualTo(liveCardinality);
    assertThat(restored.documents()).isEqualTo(10);
    assertThat(restored.occurrences()).isEqualTo(80);
    assertThat(restored.estimate("court")).isEqualTo(50);
    assertThat(restored.estimate("ruling")).isEqualTo(20);
    assertThat(restored.estimate("appeal")).isEqualTo(10);
  }

  @Test
  void explicitSnapshotOnAnEmptyWindowPersistsNothing() throws Exception {
    listener = new VocabularyListener(vocabDir, 1_000, 16, null);
    assertThat(listener.snapshotNow()).isNull();
    assertThat(listener.snapshots()).isEmpty();
  }

  @Test
  void disjointWindowsHaveNoveltyOne() throws Exception {
    listener = new VocabularyListener(vocabDir, 1_000, 16, null);
    for (int doc = 0; doc < 5; doc++) {
      listener.feed(terms("alpha", 2, "beta", 1), List.of("alpha", "beta"));
    }
    final String older = listener.snapshotNow().getName();
    for (int doc = 0; doc < 5; doc++) {
      listener.feed(terms("gamma", 2, "delta", 1), List.of("gamma", "delta"));
    }

    final Map<VocabChannel, VocabularyListener.ChannelDrift> drift =
        byChannel(listener.drift(older, "live"));

    assertThat(drift.get(VocabChannel.VOCAB_CHANNEL_TERMS).metrics().noveltyRate())
        .isGreaterThan(0.9);
    // Disjoint vocabularies are also maximally divergent distributions.
    assertThat(drift.get(VocabChannel.VOCAB_CHANNEL_TERMS).metrics()
        .jensenShannonDivergence()).isGreaterThan(0.9);
  }

  @Test
  void identicalDistributionsHaveNoDrift() throws Exception {
    listener = new VocabularyListener(vocabDir, 1_000, 16, null);
    for (int doc = 0; doc < 5; doc++) {
      listener.feed(terms("alpha", 2, "beta", 1), List.of("alpha", "beta"));
    }
    final String older = listener.snapshotNow().getName();
    for (int doc = 0; doc < 5; doc++) {
      listener.feed(terms("alpha", 2, "beta", 1), List.of("alpha", "beta"));
    }

    final DriftMetrics.Result terms =
        byChannel(listener.drift(older, "live")).get(VocabChannel.VOCAB_CHANNEL_TERMS)
            .metrics();

    assertThat(terms.noveltyRate()).isLessThan(0.05);
    assertThat(terms.jensenShannonDivergence()).isLessThan(0.01);
  }

  @Test
  void shiftedFrequenciesHavePositiveDivergenceButNoNovelty() throws Exception {
    listener = new VocabularyListener(vocabDir, 1_000, 16, null);
    for (int doc = 0; doc < 5; doc++) {
      listener.feed(terms("alpha", 10, "beta", 1), List.of("alpha", "beta"));
    }
    final String older = listener.snapshotNow().getName();
    for (int doc = 0; doc < 5; doc++) {
      listener.feed(terms("alpha", 1, "beta", 10), List.of("alpha", "beta"));
    }

    final DriftMetrics.Result terms =
        byChannel(listener.drift(older, "live")).get(VocabChannel.VOCAB_CHANNEL_TERMS)
            .metrics();

    assertThat(terms.noveltyRate()).isLessThan(0.05);
    assertThat(terms.jensenShannonDivergence()).isGreaterThan(0.05);
  }

  @Test
  void embeddingCoverageReadsTheModelVocabulary() throws Exception {
    final Path embeddingsDir = vocabDir.resolve("embeddings");
    Files.createDirectories(embeddingsDir);
    Files.writeString(embeddingsDir.resolve("vocab.txt"), "alpha\nbeta\n");
    listener = new VocabularyListener(vocabDir, 1_000, 16, embeddingsDir);
    for (int doc = 0; doc < 5; doc++) {
      // Covered: alpha x2, beta x1 per doc. OOV: gamma x1 per doc.
      listener.feed(terms("alpha", 2, "beta", 1, "gamma", 1),
          List.of("alpha", "alpha", "beta", "gamma"));
    }
    final String older = listener.snapshotNow().getName();
    for (int doc = 0; doc < 5; doc++) {
      listener.feed(terms("alpha", 2, "beta", 1, "gamma", 1),
          List.of("alpha", "alpha", "beta", "gamma"));
    }

    final Map<VocabChannel, VocabularyListener.ChannelDrift> drift =
        byChannel(listener.drift(older, "live"));
    final DriftMetrics.Result tokens =
        drift.get(VocabChannel.VOCAB_CHANNEL_TOKENS).metrics();

    assertThat(tokens.embeddingOovComputed()).isTrue();
    // One of four token occurrences per document is OOV.
    assertThat(tokens.embeddingOovShare()).isCloseTo(0.25,
        org.assertj.core.data.Offset.offset(0.01));
    // Coverage is a TOKENS property; TERMS never reports it.
    assertThat(drift.get(VocabChannel.VOCAB_CHANNEL_TERMS).metrics()
        .embeddingOovComputed()).isFalse();
  }

  @Test
  void tokenizerJsonVocabIsReadToo() throws Exception {
    final Path embeddingsDir = vocabDir.resolve("embeddings");
    Files.createDirectories(embeddingsDir);
    Files.writeString(embeddingsDir.resolve("tokenizer.json"),
        "{\"model\": {\"type\": \"BPE\", \"vocab\": {\"alpha\": 0, \"be\\u00e4\": 1}}}");
    listener = new VocabularyListener(vocabDir, 1_000, 16, embeddingsDir);
    listener.feed(terms("alpha", 1), List.of("alpha", "gamma"));

    final DriftMetrics.Result tokens =
        byChannel(listener.drift("live", "live")).get(VocabChannel.VOCAB_CHANNEL_TOKENS)
            .metrics();

    assertThat(tokens.embeddingOovComputed()).isTrue();
    assertThat(tokens.embeddingOovShare()).isCloseTo(0.5,
        org.assertj.core.data.Offset.offset(0.01));
  }

  @Test
  void unknownSnapshotReferencesAreRejected() throws Exception {
    listener = new VocabularyListener(vocabDir, 1_000, 16, null);
    listener.feed(terms("alpha", 1), List.of("alpha"));

    org.assertj.core.api.Assertions.assertThatThrownBy(
        () -> listener.drift("snapshot-99-1.pb", "live"))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("unknown snapshot");
  }

  @Test
  void restartResumesTheSnapshotHistory() throws Exception {
    listener = new VocabularyListener(vocabDir, 1_000, 16, null);
    listener.feed(terms("alpha", 1), List.of("alpha"));
    listener.snapshotNow();
    listener.close();

    final VocabularyListener resumed =
        new VocabularyListener(vocabDir, 1_000, 16, null);
    try {
      assertThat(resumed.snapshots()).hasSize(1);
      assertThat(resumed.snapshots().get(0).getDocuments()).isEqualTo(1);
      // The next snapshot continues the sequence rather than clobbering.
      resumed.feed(terms("beta", 1), List.of("beta"));
      assertThat(resumed.snapshotNow().getSequence()).isEqualTo(1);
    } finally {
      resumed.close();
    }
  }
}
