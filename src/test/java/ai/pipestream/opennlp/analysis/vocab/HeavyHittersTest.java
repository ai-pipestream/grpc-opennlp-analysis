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

import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;

import org.junit.jupiter.api.Test;

/**
 * Space-saving top-K: the true heavy hitters survive the churn of a long
 * tail, with near-exact counts.
 */
class HeavyHittersTest {

  @Test
  void trueHeavyHittersSurviveTheTail() {
    final HeavyHitters hitters = new HeavyHitters(16);
    // Ten genuinely heavy terms, each interleaved with a long tail of
    // one-off terms that churns the eviction path.
    final String[] heavy = new String[10];
    for (int h = 0; h < 10; h++) {
      heavy[h] = "heavy-" + h;
    }
    for (int round = 0; round < 1_000; round++) {
      for (String term : heavy) {
        hitters.add(term, 1);
      }
      hitters.add("tail-" + round, 1);
    }

    final List<HeavyHitters.Entry> snapshot = hitters.snapshot();
    assertThat(snapshot).hasSize(16);
    final Map<String, Long> counts = snapshot.stream()
        .collect(Collectors.toMap(HeavyHitters.Entry::term,
            HeavyHitters.Entry::count));
    for (String term : heavy) {
      // Space-saving overestimates by at most the minimum count in the
      // table; at 1000 true occurrences the heavy terms are near-exact.
      assertThat(counts).containsKey(term);
      assertThat(counts.get(term)).isBetween(1_000L, 1_050L);
    }
    // Ordered by count descending: the heavy block leads.
    for (int i = 0; i < heavy.length; i++) {
      assertThat(snapshot.get(i).count()).isGreaterThanOrEqualTo(1_000L);
    }
  }

  @Test
  void trackedTermsAccumulateExactly() {
    final HeavyHitters hitters = new HeavyHitters(4);
    hitters.add("alpha", 3);
    hitters.add("alpha", 4);
    hitters.add("beta", 2);

    assertThat(hitters.estimate("alpha")).isEqualTo(7);
    assertThat(hitters.estimate("beta")).isEqualTo(2);
    assertThat(hitters.estimate("gamma")).isZero();
    assertThat(hitters.snapshot().stream()
        .map(HeavyHitters.Entry::term)).containsExactly("alpha", "beta");
  }

  @Test
  void restoreRebuildsTheList() {
    final HeavyHitters hitters = new HeavyHitters(8);
    hitters.add("alpha", 5);
    final List<HeavyHitters.Entry> snapshot = hitters.snapshot();

    final HeavyHitters restored = new HeavyHitters(8);
    restored.restore(snapshot);

    assertThat(restored.estimate("alpha")).isEqualTo(5);
    assertThat(restored.snapshot()).isEqualTo(snapshot);
  }

  @Test
  void entriesAreComparableRecords() {
    final Function<HeavyHitters.Entry, String> term = HeavyHitters.Entry::term;
    assertThat(term.apply(new HeavyHitters.Entry("x", 1))).isEqualTo("x");
  }
}
