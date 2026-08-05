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

import org.junit.jupiter.api.Test;

/**
 * HyperLogLog cardinality accuracy: a few percent at 10k and 100k distinct,
 * plus the merge contract the novelty rate relies on.
 */
class HyperLogLogTest {

  private static HyperLogLog filled(String prefix, int distinct) {
    final HyperLogLog sketch = new HyperLogLog();
    for (int i = 0; i < distinct; i++) {
      sketch.add(prefix + i);
    }
    return sketch;
  }

  @Test
  void emptySketchEstimatesZero() {
    assertThat(new HyperLogLog().estimate()).isEqualTo(0.0);
  }

  @Test
  void cardinalityAt10kDistinct() {
    final double estimate = filled("term-", 10_000).estimate();
    assertThat(estimate / 10_000.0).isBetween(0.95, 1.05);
  }

  @Test
  void cardinalityAt100kDistinct() {
    final double estimate = filled("term-", 100_000).estimate();
    assertThat(estimate / 100_000.0).isBetween(0.95, 1.05);
  }

  @Test
  void duplicatesDoNotCount() {
    final HyperLogLog sketch = new HyperLogLog();
    for (int i = 0; i < 1_000; i++) {
      sketch.add("term-" + (i % 100));
    }
    assertThat(sketch.estimate() / 100.0).isBetween(0.90, 1.10);
  }

  @Test
  void mergeEstimatesTheUnion() {
    // 60k + 60k distinct terms with a 20k overlap: union is 100k.
    final HyperLogLog left = filled("term-", 60_000);
    final HyperLogLog right = new HyperLogLog();
    for (int i = 40_000; i < 100_000; i++) {
      right.add("term-" + i);
    }

    left.merge(right);

    assertThat(left.estimate() / 100_000.0).isBetween(0.95, 1.05);
  }

  @Test
  void snapshotRoundTripRestoresIdenticalEstimate() {
    final HyperLogLog sketch = filled("term-", 10_000);

    final HyperLogLog restored =
        HyperLogLog.fromBytes(sketch.precision(), sketch.toBytes());

    assertThat(restored.estimate()).isEqualTo(sketch.estimate());
  }
}
