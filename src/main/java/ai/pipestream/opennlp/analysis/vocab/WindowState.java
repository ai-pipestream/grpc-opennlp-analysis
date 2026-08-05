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

import java.util.List;

import ai.pipestream.opennlp.analysis.v1.VocabChannel;
import ai.pipestream.opennlp.analysis.v1.VocabChannelSnapshot;
import ai.pipestream.opennlp.analysis.v1.VocabHeavyHitter;
import com.google.protobuf.ByteString;

/**
 * The per-channel state of one vocabulary window: the three sketches
 * (HyperLogLog for cardinality, count-min for point frequencies, space-saving
 * for the heavy hitters) plus the document and occurrence counters.
 *
 * <p>All mutating and reading methods are synchronized: AnalyzeStream feeds
 * the listener from a worker pool, and a drift comparison reading the live
 * window must never observe a half-applied document.</p>
 */
public final class WindowState {

  private final VocabChannel channel;
  private CountMinSketch countMin;
  private HyperLogLog hyperLogLog;
  private final HeavyHitters heavyHitters;
  private long documents;
  private long occurrences;

  /**
   * @param channel the channel this state belongs to, must not be {@code null}
   * @param topK the heavy-hitter list size
   */
  public WindowState(VocabChannel channel, int topK) {
    if (channel == null) {
      throw new IllegalArgumentException("channel must not be null");
    }
    this.channel = channel;
    this.countMin = new CountMinSketch();
    this.hyperLogLog = new HyperLogLog();
    this.heavyHitters = new HeavyHitters(topK);
  }

  /**
   * Adds one term with its frequency from one document.
   *
   * @param term the term, must not be {@code null}
   * @param frequency occurrences of the term in the document, must be positive
   */
  public synchronized void addTerm(String term, long frequency) {
    countMin.add(term, frequency);
    hyperLogLog.add(term);
    heavyHitters.add(term, frequency);
    occurrences += frequency;
  }

  /** Counts one more document into the window. */
  public synchronized void addDocument() {
    documents++;
  }

  /**
   * @return the channel this state belongs to
   */
  public VocabChannel channel() {
    return channel;
  }

  /**
   * @return documents accumulated in this window
   */
  public synchronized long documents() {
    return documents;
  }

  /**
   * @return total occurrences (with multiplicity) accumulated in this window
   */
  public synchronized long occurrences() {
    return occurrences;
  }

  /**
   * @return the estimated distinct-term cardinality of this window
   */
  public synchronized double cardinalityEstimate() {
    return hyperLogLog.estimate();
  }

  /**
   * @return the heavy hitters, ordered by count descending
   */
  public synchronized List<HeavyHitters.Entry> heavyHitters() {
    return heavyHitters.snapshot();
  }

  /**
   * The estimated frequency of {@code term}: its heavy-hitter count when it
   * is on the list, otherwise the count-min point query. This is the count a
   * drift comparison uses for every term on the union of the two top-K
   * lists.
   *
   * @param term the term, must not be {@code null}
   * @return the best available frequency estimate
   */
  public synchronized long estimate(String term) {
    final long heavy = heavyHitters.estimate(term);
    return heavy > 0 ? heavy : countMin.estimate(term);
  }

  /**
   * @return a detached copy of the cardinality sketch, for union computations
   */
  public synchronized HyperLogLog cardinalitySketch() {
    return hyperLogLog.copy();
  }

  /**
   * Serializes this window's channel state into its persisted form.
   *
   * @return the channel snapshot message
   */
  public synchronized VocabChannelSnapshot toProto() {
    final VocabChannelSnapshot.Builder out = VocabChannelSnapshot.newBuilder()
        .setChannel(channel)
        .setDocuments(documents)
        .setTermOccurrences(occurrences)
        .setHllPrecision(hyperLogLog.precision())
        .setHllRegisters(ByteString.copyFrom(hyperLogLog.toBytes()))
        .setCmsDepth(countMin.depth())
        .setCmsWidth(countMin.width())
        .setCmsTable(ByteString.copyFrom(countMin.toBytes()));
    for (HeavyHitters.Entry entry : heavyHitters.snapshot()) {
      out.addHeavyHitters(VocabHeavyHitter.newBuilder()
          .setTerm(entry.term()).setCount(entry.count()));
    }
    return out.build();
  }

  /**
   * Restores a window's channel state from its persisted form.
   *
   * @param snapshot the persisted channel state, must not be {@code null}
   * @return the restored window state
   */
  public static WindowState fromProto(VocabChannelSnapshot snapshot) {
    if (snapshot == null) {
      throw new IllegalArgumentException("snapshot must not be null");
    }
    final WindowState state =
        new WindowState(snapshot.getChannel(), Math.max(1, snapshot.getHeavyHittersCount()));
    state.documents = snapshot.getDocuments();
    state.occurrences = snapshot.getTermOccurrences();
    state.hyperLogLog = HyperLogLog.fromBytes(snapshot.getHllPrecision(),
        snapshot.getHllRegisters().toByteArray());
    state.countMin = CountMinSketch.fromBytes(snapshot.getCmsDepth(), snapshot.getCmsWidth(),
        snapshot.getCmsTable().toByteArray(), snapshot.getTermOccurrences());
    state.heavyHitters.restore(snapshot.getHeavyHittersList().stream()
        .map(h -> new HeavyHitters.Entry(h.getTerm(), h.getCount()))
        .toList());
    return state;
  }
}
