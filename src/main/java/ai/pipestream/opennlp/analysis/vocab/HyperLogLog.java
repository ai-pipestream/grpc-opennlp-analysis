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

/**
 * A HyperLogLog cardinality sketch over 64-bit hashes.
 *
 * <p>Precision p = 14 → 16384 one-byte registers (16 KiB), standard error
 * ~0.8%. Distinct-term cardinality is the metric behind the novelty rate:
 * registers merge by cell-wise maximum, so the union cardinality of two
 * windows comes from merging their sketches, which is what makes
 * {@code (|A ∪ B| − |A|) / |B|} computable without ever materializing the
 * vocabulary itself.</p>
 *
 * <p>Not thread-safe; the owning window state serializes access.</p>
 */
public final class HyperLogLog {

  /** Sketch precision: log2 of the register count. */
  public static final int PRECISION = 14;

  private final int precision;
  private final int registerCount;
  private final byte[] registers;

  /** A fresh, empty sketch of the standard precision. */
  public HyperLogLog() {
    this(PRECISION);
  }

  /**
   * @param precision log2 of the register count, between 4 and 16
   */
  public HyperLogLog(int precision) {
    if (precision < 4 || precision > 16) {
      throw new IllegalArgumentException("precision must be in [4, 16]");
    }
    this.precision = precision;
    this.registerCount = 1 << precision;
    this.registers = new byte[registerCount];
  }

  /**
   * Adds one occurrence of {@code term}. Repeated adds of the same term are
   * idempotent up to hash collisions.
   *
   * @param term the term, must not be {@code null}
   */
  public void add(String term) {
    if (term == null) {
      throw new IllegalArgumentException("term must not be null");
    }
    final long hash = Hashes.base(term);
    // The top p bits choose the register; the rank is the number of leading
    // zeros of the remaining 64 - p bits, plus one.
    final int index = (int) (hash >>> (Long.SIZE - precision));
    final int rank = Long.numberOfLeadingZeros((hash << precision) | (1L << (precision - 1))) + 1;
    if (rank > registers[index]) {
      registers[index] = (byte) rank;
    }
  }

  /**
   * The estimated number of distinct terms added. Small cardinalities use
   * linear counting (the raw estimator is badly biased below ~2.5 registers
   * per occupied bucket); 64-bit hashes make the large-range correction
   * irrelevant at any corpus scale this service will see.
   *
   * @return the estimated cardinality
   */
  public double estimate() {
    double inverseSum = 0.0;
    int zeros = 0;
    for (byte register : registers) {
      final int value = register;
      inverseSum += Math.pow(2.0, -value);
      if (value == 0) {
        zeros++;
      }
    }
    final double alpha = 0.7213 / (1.0 + 1.079 / registerCount);
    final double raw = alpha * registerCount * registerCount / inverseSum;
    if (raw <= 2.5 * registerCount && zeros > 0) {
      return registerCount * Math.log((double) registerCount / zeros);
    }
    return raw;
  }

  /**
   * Merges {@code other} into this sketch, cell-wise maximum. Only meaningful
   * for sketches of the same precision.
   *
   * @param other the sketch to fold in, must not be {@code null}
   */
  public void merge(HyperLogLog other) {
    if (other == null) {
      throw new IllegalArgumentException("other must not be null");
    }
    if (other.precision != precision) {
      throw new IllegalArgumentException("precisions differ: " + other.precision
          + " vs " + precision);
    }
    for (int i = 0; i < registers.length; i++) {
      if (other.registers[i] > registers[i]) {
        registers[i] = other.registers[i];
      }
    }
  }

  /**
   * @return a sketch with the same registers and precision, detached from this one
   */
  public HyperLogLog copy() {
    return fromBytes(precision, registers.clone());
  }

  /**
   * @return the precision this sketch was built with
   */
  public int precision() {
    return precision;
  }

  /**
   * Serializes the registers in register order — the {@code hll_registers}
   * encoding of {@code VocabChannelSnapshot}.
   *
   * @return a copy of the register bytes, one per register
   */
  public byte[] toBytes() {
    return registers.clone();
  }

  /**
   * Restores a sketch from {@link #toBytes()} output.
   *
   * @param precision log2 of the register count
   * @param bytes the register bytes, must not be {@code null}
   * @return the restored sketch
   */
  public static HyperLogLog fromBytes(int precision, byte[] bytes) {
    if (bytes == null) {
      throw new IllegalArgumentException("bytes must not be null");
    }
    final HyperLogLog sketch = new HyperLogLog(precision);
    if (bytes.length != sketch.registerCount) {
      throw new IllegalArgumentException("register byte length " + bytes.length
          + " does not match precision " + precision);
    }
    System.arraycopy(bytes, 0, sketch.registers, 0, bytes.length);
    return sketch;
  }
}
