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
 * A count-min sketch: approximate per-term frequencies with a one-sided
 * error (estimates are never below the true count).
 *
 * <p>Depth 5, width 2<sup>17</sup> — five independent hash functions over a
 * 131072-column table of {@code long} counters (5 MiB). Point queries answer
 * "how often did this term occur" for terms missing from the heavy-hitter
 * list; the drift metrics use exactly that to fill the tail of a top-K
 * comparison. Sketches merge cell-wise, so cross-instance and cross-window
 * combination is a tooling problem with a trivial answer.</p>
 *
 * <p>Not thread-safe; the owning window state serializes access.</p>
 */
public final class CountMinSketch {

  /** Sketch depth: the number of rows (independent hash functions). */
  public static final int DEPTH = 5;

  /** Sketch width: columns per row, a power of two for cheap indexing. */
  public static final int WIDTH = 1 << 17;

  private final int depth;
  private final int width;
  private final long[] table;
  private long total;

  /** A fresh, empty sketch of the standard shape. */
  public CountMinSketch() {
    this(DEPTH, WIDTH);
  }

  /**
   * @param depth number of rows, must be positive
   * @param width columns per row, must be a positive power of two
   */
  public CountMinSketch(int depth, int width) {
    if (depth <= 0 || width <= 0 || (width & (width - 1)) != 0) {
      throw new IllegalArgumentException(
          "depth must be positive and width must be a positive power of two");
    }
    this.depth = depth;
    this.width = width;
    this.table = new long[depth * width];
  }

  /**
   * Adds {@code count} occurrences of {@code term}.
   *
   * @param term the term, must not be {@code null}
   * @param count occurrences to add, must not be negative
   */
  public void add(String term, long count) {
    if (term == null) {
      throw new IllegalArgumentException("term must not be null");
    }
    if (count < 0) {
      throw new IllegalArgumentException("count must not be negative");
    }
    if (count == 0) {
      return;
    }
    final long base = Hashes.base(term);
    for (int row = 0; row < depth; row++) {
      final int column = (int) (Hashes.row(base, row) & (width - 1));
      table[row * width + column] += count;
    }
    total += count;
  }

  /**
   * The estimated frequency of {@code term}: the minimum over the rows.
   * Never below the true count; within the sketch's error bound above it.
   *
   * @param term the term, must not be {@code null}
   * @return the estimated frequency
   */
  public long estimate(String term) {
    if (term == null) {
      throw new IllegalArgumentException("term must not be null");
    }
    final long base = Hashes.base(term);
    long min = Long.MAX_VALUE;
    for (int row = 0; row < depth; row++) {
      final int column = (int) (Hashes.row(base, row) & (width - 1));
      min = Math.min(min, table[row * width + column]);
    }
    return min;
  }

  /**
   * Merges {@code other} into this sketch, cell-wise. Only meaningful for
   * sketches of the same shape.
   *
   * @param other the sketch to fold in, must not be {@code null}
   */
  public void merge(CountMinSketch other) {
    if (other == null) {
      throw new IllegalArgumentException("other must not be null");
    }
    if (other.depth != depth || other.width != width) {
      throw new IllegalArgumentException("sketch shapes differ: " + other.depth + "x"
          + other.width + " vs " + depth + "x" + width);
    }
    for (int i = 0; i < table.length; i++) {
      table[i] += other.table[i];
    }
    total += other.total;
  }

  /**
   * @return the total number of occurrences ever added
   */
  public long totalCount() {
    return total;
  }

  /**
   * @return the number of rows
   */
  public int depth() {
    return depth;
  }

  /**
   * @return the number of columns per row
   */
  public int width() {
    return width;
  }

  /**
   * Serializes the table in row-major order as little-endian unsigned 64-bit
   * counters — the {@code cms_table} encoding of {@code VocabChannelSnapshot}.
   *
   * @return the table bytes, {@code depth * width * 8} long
   */
  public byte[] toBytes() {
    final byte[] out = new byte[table.length * Long.BYTES];
    for (int i = 0; i < table.length; i++) {
      writeLong(out, i * Long.BYTES, table[i]);
    }
    return out;
  }

  /**
   * Restores a sketch from {@link #toBytes()} output.
   *
   * @param depth number of rows
   * @param width columns per row
   * @param bytes the serialized table, must not be {@code null}
   * @param total the total occurrence count carried alongside the table
   * @return the restored sketch
   */
  public static CountMinSketch fromBytes(int depth, int width, byte[] bytes, long total) {
    if (bytes == null) {
      throw new IllegalArgumentException("bytes must not be null");
    }
    final CountMinSketch sketch = new CountMinSketch(depth, width);
    if (bytes.length != sketch.table.length * Long.BYTES) {
      throw new IllegalArgumentException("table byte length " + bytes.length
          + " does not match " + depth + "x" + width);
    }
    for (int i = 0; i < sketch.table.length; i++) {
      sketch.table[i] = readLong(bytes, i * Long.BYTES);
    }
    sketch.total = total;
    return sketch;
  }

  private static void writeLong(byte[] out, int offset, long value) {
    for (int i = 0; i < Long.BYTES; i++) {
      out[offset + i] = (byte) (value >>> (8 * i));
    }
  }

  private static long readLong(byte[] in, int offset) {
    long value = 0;
    for (int i = 0; i < Long.BYTES; i++) {
      value |= (long) (in[offset + i] & 0xFF) << (8 * i);
    }
    return value;
  }
}
