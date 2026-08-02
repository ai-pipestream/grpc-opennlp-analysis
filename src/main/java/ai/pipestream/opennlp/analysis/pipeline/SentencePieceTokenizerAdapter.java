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

package ai.pipestream.opennlp.analysis.pipeline;

import java.util.List;

import opennlp.subword.sentencepiece.SentencePieceTokenizer;
import opennlp.tools.tokenize.SubwordPiece;
import opennlp.tools.tokenize.Tokenizer;
import opennlp.tools.util.Span;

/**
 * Adapts a {@link SentencePieceTokenizer} onto the {@link Tokenizer} interface
 * so it can serve as the pipeline's tokenizer. SentencePiece implements
 * {@code SubwordTokenizer}, not {@code Tokenizer}, so it cannot go into a
 * {@code TokenizerAnnotator} directly.
 *
 * <p>Spans are the piece spans, which SentencePiece maps back to the caller's
 * original text through the model's normalizer. Pieces with an empty span
 * (control symbols, byte-fallback fill bytes) are dropped: they carry no
 * original text and a zero-width token would break span consumers.
 * {@link #tokenize(String)} returns the covered ORIGINAL text of each span,
 * not the normalized piece string, keeping token text consistent with every
 * other tokenizer.</p>
 *
 * <p>Thread safety: the wrapped tokenizer is immutable after loading.</p>
 */
final class SentencePieceTokenizerAdapter implements Tokenizer {

  private final SentencePieceTokenizer delegate;

  SentencePieceTokenizerAdapter(SentencePieceTokenizer delegate) {
    if (delegate == null) {
      throw new IllegalArgumentException("delegate must not be null");
    }
    this.delegate = delegate;
  }

  @Override
  public Span[] tokenizePos(String s) {
    final List<SubwordPiece> pieces = delegate.encode(s);
    return pieces.stream()
        .filter(p -> p.start() < p.end())
        .map(SubwordPiece::span)
        .toArray(Span[]::new);
  }

  @Override
  public String[] tokenize(String s) {
    final Span[] spans = tokenizePos(s);
    final String[] tokens = new String[spans.length];
    for (int i = 0; i < spans.length; i++) {
      tokens[i] = spans[i].getCoveredText(s).toString();
    }
    return tokens;
  }
}
