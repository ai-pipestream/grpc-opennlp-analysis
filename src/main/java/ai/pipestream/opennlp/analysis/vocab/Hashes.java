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

import java.nio.charset.StandardCharsets;

/**
 * 64-bit hashing for the vocabulary sketches, implemented locally so the
 * listener needs no hashing library on the classpath.
 *
 * <p>The base hash is FNV-1a over the term's UTF-8 bytes, run through the
 * murmur3 64-bit finalizer for avalanche. Per-row independence (the count-min
 * sketch needs one hash function per row) comes from stepping a splitmix64
 * mixer on the base hash: cheap, deterministic, and statistically adequate
 * for sketching.</p>
 */
final class Hashes {

  /** The splitmix64 increment (the golden ratio as a 64-bit odd constant). */
  private static final long GOLDEN = 0x9E3779B97F4A7C15L;

  /** FNV-1a 64-bit offset basis. */
  private static final long FNV_OFFSET = 0xCBF29CE484222325L;

  /** FNV-1a 64-bit prime. */
  private static final long FNV_PRIME = 0x100000001B3L;

  private Hashes() {
  }

  /**
   * The base 64-bit hash of a term: FNV-1a over its UTF-8 bytes, avalanched
   * with the murmur3 finalizer.
   *
   * @param term the term to hash, must not be {@code null}
   * @return the 64-bit base hash
   */
  static long base(String term) {
    long hash = FNV_OFFSET;
    for (byte b : term.getBytes(StandardCharsets.UTF_8)) {
      hash ^= b & 0xFF;
      hash *= FNV_PRIME;
    }
    return fmix64(hash);
  }

  /**
   * The hash function of count-min row {@code row}: one splitmix64 step over
   * the base hash seeded with a per-row stride.
   *
   * @param base the term's base hash
   * @param row the sketch row
   * @return an independent 64-bit hash for that row
   */
  static long row(long base, int row) {
    return fmix64(base + GOLDEN * (row + 1L));
  }

  /** The murmur3 64-bit finalizer: full avalanche in three xorshift-multiply rounds. */
  private static long fmix64(long value) {
    value ^= value >>> 33;
    value *= 0xFF51AFD7ED558CCDL;
    value ^= value >>> 33;
    value *= 0xC4CEB9FE1A85EC53L;
    value ^= value >>> 33;
    return value;
  }
}
