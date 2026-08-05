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

import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

/**
 * The drift metrics between two vocabulary windows, computed from their
 * sketches: cardinality of both and their union, the novelty rate, the
 * Jensen-Shannon divergence over the union of the heavy-hitter lists, and
 * (when an embedding vocabulary is available) the out-of-vocabulary share of
 * the newer window.
 */
public final class DriftMetrics {

  private DriftMetrics() {
  }

  /**
   * The metrics of one channel comparison.
   *
   * @param fromCardinality estimated distinct-term count of the older window
   * @param toCardinality estimated distinct-term count of the newer window
   * @param unionCardinality estimated distinct-term count of the union
   * @param noveltyRate {@code (|union| − |from|) / |to|}, in [0, 1]
   * @param jensenShannonDivergence JS divergence (base 2) over the union of
   *                                the two heavy-hitter lists, in [0, 1]
   * @param embeddingOovComputed whether {@code embeddingOovShare} is
   *                             meaningful
   * @param embeddingOovShare share of the newer window's heavy-hitter token
   *                          mass absent from the embedding vocabulary
   */
  public record Result(double fromCardinality, double toCardinality,
                       double unionCardinality, double noveltyRate,
                       double jensenShannonDivergence,
                       boolean embeddingOovComputed, double embeddingOovShare) {
  }

  /**
   * Computes the drift of {@code to} (the newer window) relative to
   * {@code from} (the older window).
   *
   * @param from the older window, must not be {@code null}
   * @param to the newer window, must not be {@code null}
   * @param embeddingVocabulary the embedding model's vocabulary, or
   *                            {@code null} when coverage is not computable
   * @return the metrics, never {@code null}
   */
  public static Result compute(WindowState from, WindowState to,
                               Set<String> embeddingVocabulary) {
    if (from == null || to == null) {
      throw new IllegalArgumentException("from and to must not be null");
    }
    // Union cardinality by inclusion-exclusion over merged registers: the
    // whole point of HyperLogLog mergeability.
    final HyperLogLog merged = from.cardinalitySketch();
    merged.merge(to.cardinalitySketch());
    final double fromCardinality = from.cardinalityEstimate();
    final double toCardinality = to.cardinalityEstimate();
    final double unionCardinality = merged.estimate();
    final double noveltyRate = toCardinality > 0
        ? Math.max(0.0, (unionCardinality - fromCardinality) / toCardinality)
        : 0.0;
    final double divergence = jensenShannon(from, to);
    final boolean oovComputed = embeddingVocabulary != null && to.occurrences() > 0;
    final double oovShare = oovComputed ? oovShare(to, embeddingVocabulary) : 0.0;
    return new Result(fromCardinality, toCardinality, unionCardinality, noveltyRate,
        divergence, oovComputed, oovShare);
  }

  /**
   * Jensen-Shannon divergence (base 2, so bounded by 1) between the two term
   * distributions over the union of their heavy-hitter lists. Counts for a
   * term missing from one list come from that window's count-min sketch, so
   * the comparison sees the tail too, at sketch accuracy.
   */
  private static double jensenShannon(WindowState from, WindowState to) {
    final Set<String> union = new LinkedHashSet<>();
    from.heavyHitters().forEach(e -> union.add(e.term()));
    to.heavyHitters().forEach(e -> union.add(e.term()));
    long totalFrom = 0;
    long totalTo = 0;
    final long[] countsFrom = new long[union.size()];
    final long[] countsTo = new long[union.size()];
    int i = 0;
    for (String term : union) {
      countsFrom[i] = from.estimate(term);
      countsTo[i] = to.estimate(term);
      totalFrom += countsFrom[i];
      totalTo += countsTo[i];
      i++;
    }
    if (totalFrom == 0 && totalTo == 0) {
      return 0.0;
    }
    if (totalFrom == 0 || totalTo == 0) {
      // One side has no mass at all: maximal divergence.
      return 1.0;
    }
    double divergence = 0.0;
    for (int j = 0; j < countsFrom.length; j++) {
      final double p = (double) countsFrom[j] / totalFrom;
      final double q = (double) countsTo[j] / totalTo;
      final double m = (p + q) / 2.0;
      if (p > 0) {
        divergence += 0.5 * p * log2(p / m);
      }
      if (q > 0) {
        divergence += 0.5 * q * log2(q / m);
      }
    }
    return divergence;
  }

  /**
   * The out-of-vocabulary share of the window's token mass, computed over
   * its heavy-hitter list: the tail beyond top-K contributes negligible mass
   * on a realistically skewed corpus, which is the same premise the JS
   * divergence runs on.
   */
  private static double oovShare(WindowState window, Set<String> vocabulary) {
    final List<HeavyHitters.Entry> hitters = window.heavyHitters();
    long mass = 0;
    long oov = 0;
    for (HeavyHitters.Entry entry : hitters) {
      mass += entry.count();
      if (!vocabulary.contains(entry.term())) {
        oov += entry.count();
      }
    }
    return mass > 0 ? (double) oov / mass : 0.0;
  }

  private static double log2(double value) {
    return Math.log(value) / Math.log(2.0);
  }
}
