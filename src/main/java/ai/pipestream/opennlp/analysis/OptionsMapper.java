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

package ai.pipestream.opennlp.analysis;

import ai.pipestream.opennlp.analysis.pipeline.PipelineOptions;
import ai.pipestream.opennlp.analysis.v1.AnalysisOptions;
import ai.pipestream.opennlp.analysis.v1.EmbeddingOptions;
import ai.pipestream.opennlp.analysis.v1.TermVectorOptions;
import io.grpc.Status;

/**
 * Maps the proto {@link AnalysisOptions} onto the gRPC-free
 * {@link PipelineOptions}, resolving UNSPECIFIED enum values to server
 * defaults and rejecting unrecognized values with INVALID_ARGUMENT.
 */
final class OptionsMapper {

  private OptionsMapper() {
  }

  static PipelineOptions fromProto(AnalysisOptions proto) {
    final PipelineOptions.TermVectorSpec termVectors;
    if (proto.hasTermVectors() && proto.getTermVectors().getEnabled()) {
      termVectors = new PipelineOptions.TermVectorSpec(
          termVectorMode(proto.getTermVectors().getMode()),
          proto.getTermVectors().getRungsList().stream()
              .map(OptionsMapper::rung)
              .filter(java.util.Objects::nonNull)
              .toList());
    } else {
      termVectors = null;
    }
    return new PipelineOptions(
        proto.getLanguage().isBlank() ? "en" : proto.getLanguage(),
        tokenizer(proto.getTokenizer()),
        proto.getSentenceDetection(),
        proto.getPosTags(),
        proto.getNer(),
        stemmer(proto.getStemmer()),
        termVectors,
        proto.hasEmbeddings() ? embeddingSource(proto.getEmbeddings().getSource()) : null);
  }

  private static PipelineOptions.Tokenizer tokenizer(AnalysisOptions.Tokenizer value) {
    return switch (value) {
      case TOKENIZER_UNSPECIFIED, TOKENIZER_WHITESPACE -> PipelineOptions.Tokenizer.WHITESPACE;
      case TOKENIZER_SIMPLE -> PipelineOptions.Tokenizer.SIMPLE;
      case UNRECOGNIZED -> throw Status.INVALID_ARGUMENT
          .withDescription("unrecognized tokenizer value").asRuntimeException();
    };
  }

  private static PipelineOptions.Stemmer stemmer(AnalysisOptions.Stemmer value) {
    return switch (value) {
      case STEMMER_UNSPECIFIED, STEMMER_NONE -> PipelineOptions.Stemmer.NONE;
      case STEMMER_PORTER -> PipelineOptions.Stemmer.PORTER;
      case STEMMER_SNOWBALL_ENGLISH -> PipelineOptions.Stemmer.SNOWBALL_ENGLISH;
      case STEMMER_SNOWBALL_GERMAN -> PipelineOptions.Stemmer.SNOWBALL_GERMAN;
      case STEMMER_SNOWBALL_FRENCH -> PipelineOptions.Stemmer.SNOWBALL_FRENCH;
      case STEMMER_SNOWBALL_SPANISH -> PipelineOptions.Stemmer.SNOWBALL_SPANISH;
      case STEMMER_LIGHT_ENGLISH -> PipelineOptions.Stemmer.LIGHT_ENGLISH;
      case STEMMER_LIGHT_GERMAN -> PipelineOptions.Stemmer.LIGHT_GERMAN;
      case STEMMER_LIGHT_FRENCH -> PipelineOptions.Stemmer.LIGHT_FRENCH;
      case STEMMER_LIGHT_SPANISH -> PipelineOptions.Stemmer.LIGHT_SPANISH;
      case UNRECOGNIZED -> throw Status.INVALID_ARGUMENT
          .withDescription("unrecognized stemmer value").asRuntimeException();
    };
  }

  private static PipelineOptions.TermVectorMode termVectorMode(
      TermVectorOptions.Mode value) {
    return switch (value) {
      case MODE_UNSPECIFIED, MODE_FULL -> PipelineOptions.TermVectorMode.FULL;
      case MODE_SCORING_ONLY -> PipelineOptions.TermVectorMode.SCORING_ONLY;
      case UNRECOGNIZED -> throw Status.INVALID_ARGUMENT
          .withDescription("unrecognized term vector mode value").asRuntimeException();
    };
  }

  private static PipelineOptions.NormalizerRung rung(
      TermVectorOptions.NormalizerRung value) {
    return switch (value) {
      case NORMALIZER_RUNG_STRIP_INVISIBLE -> PipelineOptions.NormalizerRung.STRIP_INVISIBLE;
      case NORMALIZER_RUNG_WHITESPACE -> PipelineOptions.NormalizerRung.WHITESPACE;
      case NORMALIZER_RUNG_DASHES -> PipelineOptions.NormalizerRung.DASHES;
      case NORMALIZER_RUNG_QUOTES -> PipelineOptions.NormalizerRung.QUOTES;
      case NORMALIZER_RUNG_DIGITS -> PipelineOptions.NormalizerRung.DIGITS;
      case NORMALIZER_RUNG_FULL_CASE_FOLD -> PipelineOptions.NormalizerRung.FULL_CASE_FOLD;
      // UNSPECIFIED rungs are ignored per the contract.
      case NORMALIZER_RUNG_UNSPECIFIED -> null;
      case UNRECOGNIZED -> throw Status.INVALID_ARGUMENT
          .withDescription("unrecognized normalizer rung value").asRuntimeException();
    };
  }

  private static PipelineOptions.EmbeddingSource embeddingSource(
      EmbeddingOptions.Source value) {
    return switch (value) {
      case SOURCE_UNSPECIFIED, SOURCE_SENTENCES -> PipelineOptions.EmbeddingSource.SENTENCES;
      case SOURCE_TOKENS -> PipelineOptions.EmbeddingSource.TOKENS;
      case UNRECOGNIZED -> throw Status.INVALID_ARGUMENT
          .withDescription("unrecognized embedding source value").asRuntimeException();
    };
  }
}
