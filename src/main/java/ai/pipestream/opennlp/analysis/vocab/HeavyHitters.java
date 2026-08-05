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

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * A space-saving top-K sketch: the heaviest terms of the window with
 * near-exact counts.
 *
 * <p>When the table is full, a new term evicts the current minimum and
 * inherits its count plus its own — the classic space-saving rule, which
 * guarantees every entry's count overestimates its true count by at most the
 * minimum count in the table. This is the list the Jensen-Shannon divergence
 * is computed over and the list a Model2Vec reweight consumes: the tail
 * beyond K contributes negligible mass on any realistically skewed corpus,
 * and the count-min sketch fills its counts when a comparison needs
 * them.</p>
 *
 * <p>Not thread-safe; the owning window state serializes access.</p>
 */
public final class HeavyHitters {

  /** The default list size. */
  public static final int DEFAULT_CAPACITY = 1024;

  /** One heavy hitter: a term and its attributed count. */
  public record Entry(String term, long count) {
  }

  private final int capacity;
  private final Map<String, Long> counts = new HashMap<>();

  /**
   * @param capacity the list size K, must be positive
   */
  public HeavyHitters(int capacity) {
    if (capacity <= 0) {
      throw new IllegalArgumentException("capacity must be positive");
    }
    this.capacity = capacity;
  }

  /**
   * Adds {@code count} occurrences of {@code term}, evicting the minimum
   * entry when the table is full.
   *
   * @param term the term, must not be {@code null}
   * @param count occurrences to add, must be positive
   */
  public void add(String term, long count) {
    if (term == null) {
      throw new IllegalArgumentException("term must not be null");
    }
    if (count <= 0) {
      throw new IllegalArgumentException("count must be positive");
    }
    final Long current = counts.get(term);
    if (current != null) {
      counts.put(term, current + count);
    } else if (counts.size() < capacity) {
      counts.put(term, count);
    } else {
      // Space-saving eviction: the newcomer takes the minimum entry's slot
      // and count. A linear scan is fine at K ~ 1024 — it runs only for
      // terms the table has never seen, and hashing already dominates.
      String minTerm = null;
      long minCount = Long.MAX_VALUE;
      for (Map.Entry<String, Long> entry : counts.entrySet()) {
        if (entry.getValue() < minCount) {
          minCount = entry.getValue();
          minTerm = entry.getKey();
        }
      }
      counts.remove(minTerm);
      counts.put(term, minCount + count);
    }
  }

  /**
   * The attributed count of {@code term}, or 0 when the term is not on the
   * list. Absence means "not heavy enough to track", not "never seen" —
   * point queries belong on the count-min sketch.
   *
   * @param term the term, must not be {@code null}
   * @return the attributed count, or 0
   */
  public long estimate(String term) {
    if (term == null) {
      throw new IllegalArgumentException("term must not be null");
    }
    return counts.getOrDefault(term, 0L);
  }

  /**
   * The list, ordered by count descending (ties by term, so snapshots are
   * deterministic).
   *
   * @return the heavy hitters, at most {@code capacity} entries
   */
  public List<Entry> snapshot() {
    final List<Entry> entries = new ArrayList<>(counts.size());
    counts.forEach((term, count) -> entries.add(new Entry(term, count)));
    entries.sort((a, b) -> a.count() != b.count()
        ? Long.compare(b.count(), a.count())
        : a.term().compareTo(b.term()));
    return entries;
  }

  /**
   * @return the list size K
   */
  public int capacity() {
    return capacity;
  }

  /**
   * Restores a list from a snapshot, replacing any current content. Used
   * when a persisted window is loaded for a drift comparison.
   *
   * @param entries the snapshot entries, must not be {@code null}
   */
  public void restore(List<Entry> entries) {
    if (entries == null) {
      throw new IllegalArgumentException("entries must not be null");
    }
    counts.clear();
    for (Entry entry : entries) {
      counts.put(entry.term(), entry.count());
    }
  }
}
