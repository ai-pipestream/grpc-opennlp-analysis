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

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Stream;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ai.pipestream.opennlp.analysis.v1.VocabChannel;
import ai.pipestream.opennlp.analysis.v1.VocabChannelStats;
import ai.pipestream.opennlp.analysis.v1.VocabHeavyHitter;
import ai.pipestream.opennlp.analysis.v1.VocabSnapshot;
import ai.pipestream.opennlp.analysis.v1.VocabSnapshotDescriptor;

/**
 * The vocabulary listener: streaming statistics over every term and token the
 * analysis pipeline produces on the AnalyzeStream ingest path.
 *
 * <p>Every fed document counts into two independent channels — TERMS (the
 * term-vector entries with their frequencies, exactly the BM25 index
 * identity) and TOKENS (the raw token surface forms). Each channel keeps a
 * HyperLogLog, a count-min sketch, and a space-saving heavy-hitter list per
 * window, so memory is bounded by configuration, never by corpus size. When
 * the window reaches {@code windowDocs} documents it is sealed and persisted
 * as a protobuf {@link VocabSnapshot}; the same happens on an explicit
 * snapshot request and, best effort, on JVM shutdown.</p>
 *
 * <p>The listener is analytics, not a ledger: {@link #feed} never throws —
 * a sketch failure loses one document's counts, never the document's
 * analysis. At startup the snapshot directory is scanned and prior snapshots
 * are indexed from their metadata alone; full sketches are loaded lazily
 * when a drift comparison names them, so a reindex resumes its vocabulary
 * history instead of starting over.</p>
 */
public final class VocabularyListener {

  /** One term-vector entry as the listener consumes it: a term and its frequency. */
  public record TermFrequency(String term, long frequency) {
  }

  /** The drift metrics of one channel comparison. */
  public record ChannelDrift(VocabChannel channel, DriftMetrics.Result metrics) {
  }

  private static final Logger LOG = LoggerFactory.getLogger(VocabularyListener.class);

  /** The reference that names the live window in a drift request. */
  public static final String LIVE = "live";

  private static final Pattern SNAPSHOT_FILE = Pattern.compile("snapshot-(\\d+)-(\\d+)\\.pb");

  private final Object lock = new Object();
  private final Path vocabDir;
  private final long windowDocs;
  private final int topK;
  private final Set<String> embeddingVocabulary;
  private final List<VocabSnapshotDescriptor> snapshots = new ArrayList<>();
  private final Map<Long, Path> filesBySequence = new HashMap<>();
  private final Thread shutdownHook;

  private WindowState terms;
  private WindowState tokens;
  private long windowStartedMillis;
  private long nextSequence;

  /**
   * Creates the listener and indexes the prior snapshots in {@code vocabDir}.
   *
   * @param vocabDir the snapshot directory, must not be {@code null}; created
   *                 when missing, must be writable
   * @param windowDocs documents per window before automatic rollover
   * @param topK the heavy-hitter list size per channel
   * @param embeddingsDir the embedding model directory the coverage metric
   *                      reads its vocabulary from, or {@code null}
   * @throws IOException when the directory cannot be created, listed, or probed
   */
  public VocabularyListener(Path vocabDir, long windowDocs, int topK, Path embeddingsDir)
      throws IOException {
    if (vocabDir == null) {
      throw new IllegalArgumentException("vocabDir must not be null");
    }
    if (windowDocs <= 0) {
      throw new IllegalArgumentException("windowDocs must be positive");
    }
    if (topK <= 0) {
      throw new IllegalArgumentException("topK must be positive");
    }
    this.vocabDir = vocabDir;
    this.windowDocs = windowDocs;
    this.topK = topK;
    Files.createDirectories(vocabDir);
    probeWritable(vocabDir);
    scanSnapshots();
    this.embeddingVocabulary = EmbeddingVocabulary.load(embeddingsDir);
    this.terms = new WindowState(VocabChannel.VOCAB_CHANNEL_TERMS, topK);
    this.tokens = new WindowState(VocabChannel.VOCAB_CHANNEL_TOKENS, topK);
    this.windowStartedMillis = System.currentTimeMillis();
    this.shutdownHook = new Thread(this::persistOnShutdown, "vocab-listener-shutdown");
    Runtime.getRuntime().addShutdownHook(shutdownHook);
    LOG.info("Vocabulary listener active: dir {}, window {} docs, top-K {}, "
            + "{} prior snapshot(s), embedding vocabulary {}",
        vocabDir, windowDocs, topK, snapshots.size(),
        embeddingVocabulary != null ? "loaded" : "unavailable");
  }

  /**
   * Feeds one successfully analyzed document: its term-vector entries into
   * the TERMS channel and its raw token texts into the TOKENS channel.
   * Never throws — a listener failure must never reach ingest.
   *
   * @param termFrequencies the document's term-vector entries
   * @param tokenTexts the document's raw token surface forms
   */
  public void feed(List<TermFrequency> termFrequencies, List<String> tokenTexts) {
    try {
      synchronized (lock) {
        for (TermFrequency entry : termFrequencies) {
          if (entry.frequency() > 0) {
            terms.addTerm(entry.term(), entry.frequency());
          }
        }
        for (String token : tokenTexts) {
          tokens.addTerm(token, 1);
        }
        terms.addDocument();
        tokens.addDocument();
        if (terms.documents() >= windowDocs) {
          rolloverLocked("window size reached");
        }
      }
    } catch (RuntimeException | Error e) {
      LOG.warn("Vocabulary feed failed; the document's counts are lost, analysis "
          + "is unaffected", e);
    }
  }

  /**
   * Seals and persists the live window and starts a fresh one (the
   * {@code SnapshotVocab} rpc). An empty window is not persisted.
   *
   * @return the descriptor of the persisted snapshot, or {@code null} when the
   *         live window held no documents
   */
  public VocabSnapshotDescriptor snapshotNow() {
    synchronized (lock) {
      return rolloverLocked("explicit snapshot request");
    }
  }

  /**
   * @return documents accumulated in the live window
   */
  public long windowDocuments() {
    synchronized (lock) {
      return terms.documents();
    }
  }

  /**
   * Live-window statistics, one entry per channel.
   *
   * @return the channel stats of the live window
   */
  public List<VocabChannelStats> liveStats() {
    synchronized (lock) {
      return List.of(channelStats(terms), channelStats(tokens));
    }
  }

  /**
   * @return the persisted snapshots, ordered by sequence ascending
   */
  public List<VocabSnapshotDescriptor> snapshots() {
    synchronized (lock) {
      return List.copyOf(snapshots);
    }
  }

  /**
   * Computes the drift between two windows, per channel. Either reference
   * may be {@value #LIVE} for the current window; a named snapshot's
   * sketches are loaded from disk for the comparison.
   *
   * @param fromRef the older window: a snapshot name, a bare sequence
   *                number, or {@value #LIVE}
   * @param toRef the newer window, same rules
   * @return the per-channel drift metrics
   */
  public List<ChannelDrift> drift(String fromRef, String toRef) {
    final Map<VocabChannel, WindowState> from = resolve(fromRef);
    final Map<VocabChannel, WindowState> to = resolve(toRef);
    final List<ChannelDrift> out = new ArrayList<>(2);
    for (VocabChannel channel : List.of(VocabChannel.VOCAB_CHANNEL_TERMS,
        VocabChannel.VOCAB_CHANNEL_TOKENS)) {
      // Coverage is a TOKENS property: surface forms against the embedding
      // vocabulary. TERMS identity is folded/stemmed and not a model input.
      final Set<String> vocabulary = channel == VocabChannel.VOCAB_CHANNEL_TOKENS
          ? embeddingVocabulary : null;
      out.add(new ChannelDrift(channel,
          DriftMetrics.compute(from.get(channel), to.get(channel), vocabulary)));
    }
    return out;
  }

  /**
   * Removes the shutdown hook. For tests and orderly teardown; the server
   * lets the hook stand for the process lifetime.
   */
  public void close() {
    try {
      Runtime.getRuntime().removeShutdownHook(shutdownHook);
    } catch (IllegalStateException e) {
      // Shutdown already in progress; the hook will run on its own.
    }
  }

  /**
   * Resolves a drift reference to per-channel window state. The live window
   * resolves to the live states themselves — their readers are synchronized,
   * so a concurrent feed cannot corrupt a comparison; a named snapshot is
   * loaded from its file.
   */
  private Map<VocabChannel, WindowState> resolve(String ref) {
    if (ref == null || ref.isBlank()) {
      throw new IllegalArgumentException("snapshot reference must not be empty");
    }
    if (LIVE.equalsIgnoreCase(ref.trim())) {
      synchronized (lock) {
        return Map.of(VocabChannel.VOCAB_CHANNEL_TERMS, terms,
            VocabChannel.VOCAB_CHANNEL_TOKENS, tokens);
      }
    }
    final Path file;
    synchronized (lock) {
      file = locateSnapshot(ref.trim());
    }
    if (file == null) {
      throw new IllegalArgumentException("unknown snapshot '" + ref + "'");
    }
    final VocabSnapshot snapshot;
    try {
      snapshot = VocabSnapshot.parseFrom(Files.readAllBytes(file));
    } catch (IOException | RuntimeException e) {
      throw new IllegalArgumentException(
          "snapshot '" + ref + "' could not be read: " + e.getMessage(), e);
    }
    final Map<VocabChannel, WindowState> states = new HashMap<>(2);
    for (var channelSnapshot : snapshot.getChannelsList()) {
      states.put(channelSnapshot.getChannel(), WindowState.fromProto(channelSnapshot));
    }
    return states;
  }

  /** Finds a snapshot file by file name or bare sequence number; caller holds the lock. */
  private Path locateSnapshot(String ref) {
    try {
      final Path file = filesBySequence.get(Long.parseLong(ref));
      if (file != null) {
        return file;
      }
    } catch (NumberFormatException e) {
      // Not a bare sequence; try the file name.
    }
    for (Path file : filesBySequence.values()) {
      if (file.getFileName().toString().equals(ref)) {
        return file;
      }
    }
    return null;
  }

  /**
   * Seals the live window, persists it, and starts fresh. Returns null when
   * the window holds no documents — an empty snapshot is noise, and every
   * caller (rollover, explicit request, shutdown hook) is fine with none.
   */
  private VocabSnapshotDescriptor rolloverLocked(String reason) {
    final long documents = terms.documents();
    if (documents == 0) {
      return null;
    }
    final long sealedMillis = System.currentTimeMillis();
    final VocabSnapshot snapshot = VocabSnapshot.newBuilder()
        .setSequence(nextSequence)
        .setStartedEpochMillis(windowStartedMillis)
        .setSealedEpochMillis(sealedMillis)
        .addChannels(terms.toProto())
        .addChannels(tokens.toProto())
        .build();
    final String fileName = "snapshot-" + nextSequence + "-" + sealedMillis + ".pb";
    final Path target = vocabDir.resolve(fileName);
    try {
      // Write-then-move: a crash mid-write must never leave a half snapshot
      // that startup would later fail to index.
      final Path temp = vocabDir.resolve(fileName + ".tmp");
      Files.write(temp, snapshot.toByteArray());
      Files.move(temp, target, StandardCopyOption.ATOMIC_MOVE,
          StandardCopyOption.REPLACE_EXISTING);
    } catch (IOException | RuntimeException e) {
      LOG.error("Could not persist vocabulary snapshot {}; the window continues "
          + "instead of rolling over", target, e);
      return null;
    }
    final VocabSnapshotDescriptor descriptor = descriptorFor(snapshot, fileName,
        documents, target);
    snapshots.add(descriptor);
    filesBySequence.put(nextSequence, target);
    LOG.info("Vocabulary snapshot {} sealed ({} documents, {})", fileName, documents,
        reason);
    nextSequence++;
    terms = new WindowState(VocabChannel.VOCAB_CHANNEL_TERMS, topK);
    tokens = new WindowState(VocabChannel.VOCAB_CHANNEL_TOKENS, topK);
    windowStartedMillis = sealedMillis;
    return descriptor;
  }

  private void persistOnShutdown() {
    try {
      synchronized (lock) {
        rolloverLocked("JVM shutdown");
      }
    } catch (RuntimeException | Error e) {
      LOG.warn("Vocabulary snapshot on shutdown failed", e);
    }
  }

  /**
   * Indexes the prior snapshots in the directory: metadata only, so the
   * sketches are parsed and immediately dropped; they are loaded again when
   * a drift request names the snapshot. An unreadable snapshot is skipped
   * with a warning — one corrupt file must not disable the listener.
   */
  private void scanSnapshots() throws IOException {
    final List<VocabSnapshotDescriptor> found = new ArrayList<>();
    final Map<Long, Path> files = new HashMap<>();
    try (Stream<Path> listing = Files.list(vocabDir)) {
      for (Path file : listing.filter(Files::isRegularFile).toList()) {
        final Matcher matcher = SNAPSHOT_FILE.matcher(file.getFileName().toString());
        if (!matcher.matches()) {
          continue;
        }
        try {
          final VocabSnapshot snapshot = VocabSnapshot.parseFrom(Files.readAllBytes(file));
          final long documents = snapshot.getChannelsList().stream()
              .mapToLong(c -> c.getDocuments()).max().orElse(0);
          found.add(descriptorFor(snapshot, file.getFileName().toString(), documents,
              file));
          files.put(snapshot.getSequence(), file);
        } catch (IOException | RuntimeException e) {
          LOG.warn("Skipping unreadable vocabulary snapshot {}: {}", file,
              e.getMessage());
        }
      }
    }
    found.sort(Comparator.comparingLong(VocabSnapshotDescriptor::getSequence));
    snapshots.addAll(found);
    filesBySequence.putAll(files);
    nextSequence = found.stream()
        .mapToLong(VocabSnapshotDescriptor::getSequence).max().orElse(-1) + 1;
  }

  private static VocabSnapshotDescriptor descriptorFor(VocabSnapshot snapshot,
                                                       String fileName, long documents,
                                                       Path file) {
    long size = 0;
    try {
      size = Files.size(file);
    } catch (IOException e) {
      // Best effort; the descriptor is still useful at size 0.
    }
    return VocabSnapshotDescriptor.newBuilder()
        .setName(fileName)
        .setSequence(snapshot.getSequence())
        .setStartedEpochMillis(snapshot.getStartedEpochMillis())
        .setSealedEpochMillis(snapshot.getSealedEpochMillis())
        .setDocuments(documents)
        .setSizeBytes(size)
        .build();
  }

  private static VocabChannelStats channelStats(WindowState state) {
    final VocabChannelStats.Builder out = VocabChannelStats.newBuilder()
        .setChannel(state.channel())
        .setDocuments(state.documents())
        .setTermOccurrences(state.occurrences())
        .setCardinalityEstimate(state.cardinalityEstimate());
    for (HeavyHitters.Entry entry : state.heavyHitters()) {
      out.addHeavyHitters(VocabHeavyHitter.newBuilder()
          .setTerm(entry.term()).setCount(entry.count()));
    }
    return out.build();
  }

  /** A cheap writability probe: the listener degrades to disabled when the dir rejects it. */
  private static void probeWritable(Path dir) throws IOException {
    final Path probe = dir.resolve(".vocab-write-probe");
    Files.write(probe, new byte[0]);
    Files.delete(probe);
  }
}
