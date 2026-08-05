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

import java.util.HashMap;
import java.util.Map;

import org.junit.jupiter.api.Test;

/**
 * Count-min sketch accuracy: the one-sided guarantee (estimates never below
 * the true count) and the error bound on realistically skewed data.
 */
class CountMinSketchTest {

  /** A Zipf-ish stream: term i occurs ~1/i of the lead term's count. */
  private static Map<String, Long> skewedStream(CountMinSketch sketch, int distinct,
                                                long leadCount) {
    final Map<String, Long> truth = new HashMap<>();
    for (int i = 1; i <= distinct; i++) {
      final String term = "term-" + i;
      final long count = Math.max(1, leadCount / i);
      truth.put(term, count);
      for (long c = 0; c < count; c++) {
        sketch.add(term, 1);
      }
    }
    return truth;
  }

  @Test
  void estimatesNeverUndershootAndStayCloseForHeavyTerms() {
    final CountMinSketch sketch = new CountMinSketch();
    final Map<String, Long> truth = skewedStream(sketch, 2_000, 10_000);

    truth.forEach((term, count) ->
        assertThat(sketch.estimate(term)).as("estimate of %s", term)
            .isGreaterThanOrEqualTo(count));
    // Heavy terms sit far above the collision noise floor; their estimates
    // should be within a few percent of the truth.
    for (int i = 1; i <= 20; i++) {
      final long count = truth.get("term-" + i);
      assertThat((double) sketch.estimate("term-" + i) / count)
          .as("relative error of term-%d", i)
          .isBetween(1.0, 1.05);
    }
    // A term that never occurred still gets the one-sided estimate of the
    // noise floor, not a negative number.
    assertThat(sketch.estimate("never-seen")).isGreaterThanOrEqualTo(0);
  }

  @Test
  void mergeCombinesAsIfOneSketchSawEverything() {
    final CountMinSketch left = new CountMinSketch();
    final CountMinSketch right = new CountMinSketch();
    final CountMinSketch all = new CountMinSketch();
    for (int i = 0; i < 500; i++) {
      left.add("term-" + i, i + 1);
      right.add("term-" + i, 2L * (i + 1));
      all.add("term-" + i, 3L * (i + 1));
    }
    left.merge(right);

    for (int i = 0; i < 500; i++) {
      assertThat(left.estimate("term-" + i)).isEqualTo(all.estimate("term-" + i));
    }
    assertThat(left.totalCount()).isEqualTo(all.totalCount());
  }

  @Test
  void snapshotRoundTripRestoresIdenticalEstimates() {
    final CountMinSketch sketch = new CountMinSketch();
    final Map<String, Long> truth = skewedStream(sketch, 200, 5_000);

    final CountMinSketch restored = CountMinSketch.fromBytes(
        sketch.depth(), sketch.width(), sketch.toBytes(), sketch.totalCount());

    assertThat(restored.totalCount()).isEqualTo(sketch.totalCount());
    truth.forEach((term, count) ->
        assertThat(restored.estimate(term)).isEqualTo(sketch.estimate(term)));
  }
}
