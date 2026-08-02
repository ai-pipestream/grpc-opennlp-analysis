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

import java.util.List;
import java.util.Objects;

import ai.pipestream.opennlp.analysis.pipeline.PipelineOptions;
import ai.pipestream.opennlp.analysis.v1.AnalysisOptions;
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
    final PipelineOptions.Stemmer stemmer = stemmer(proto.getStemmer());
    final PipelineOptions.TermVectorSpec termVectors;
    if (proto.hasTermVectors() && proto.getTermVectors().getEnabled()) {
      final PipelineOptions.TermVectorSource source =
          termVectorSource(proto.getTermVectors().getSource());
      if ((source == PipelineOptions.TermVectorSource.STEMS
          || source == PipelineOptions.TermVectorSource.NORMALIZED_STEMS)
          && stemmer == PipelineOptions.Stemmer.NONE) {
        throw Status.INVALID_ARGUMENT
            .withDescription("term vector source STEMS requires a stemmer other "
                + "than STEMMER_NONE: the stem is the term identity in this mode")
            .asRuntimeException();
      }
      termVectors = new PipelineOptions.TermVectorSpec(
          termVectorMode(proto.getTermVectors().getMode()),
          proto.getTermVectors().getStepsList().stream()
              .map(OptionsMapper::step)
              .filter(Objects::nonNull)
              .toList(),
          source);
    } else {
      termVectors = null;
    }
    final PipelineOptions.GlossarySpec glossary;
    if (proto.hasGlossary()) {
      if (proto.getGlossary().getEntriesList().isEmpty()) {
        throw Status.INVALID_ARGUMENT
            .withDescription("glossary requires at least one entry")
            .asRuntimeException();
      }
      glossary = new PipelineOptions.GlossarySpec(
          proto.getGlossary().getEntriesList().stream()
              .map(e -> new opennlp.tools.glossary.GlossaryEntry(e.getId(), e.getTerm()))
              .toList(),
          proto.getGlossary().getIgnoreCase());
    } else {
      glossary = null;
    }
    final List<PipelineOptions.RelationPatternSpec> relations;
    if (proto.hasRelations()) {
      if (proto.getRelations().getPatternsList().isEmpty()) {
        throw Status.INVALID_ARGUMENT
            .withDescription("relations requires at least one pattern")
            .asRuntimeException();
      }
      relations = proto.getRelations().getPatternsList().stream()
          .map(p -> new PipelineOptions.RelationPatternSpec(
              p.getType(), p.getPath(), p.getTrigger().isEmpty() ? null : p.getTrigger()))
          .toList();
    } else {
      relations = List.of();
    }
    return new PipelineOptions(
        proto.getLanguage().isBlank() ? "en" : proto.getLanguage(),
        tokenizer(proto.getTokenizer()),
        proto.getSentenceDetection(),
        proto.getPosTags(),
        proto.getNer(),
        proto.getLemmatize(),
        stemmer,
        termVectors,
        proto.hasEmbeddings() ? embeddingSource(proto.getEmbeddings().getSource()) : null,
        proto.getNoise(),
        proto.getArtifacts(),
        glossary,
        proto.getPii(),
        proto.getCoref(),
        proto.getDependencyParse(),
        proto.getGeo(),
        relations);
  }

  private static PipelineOptions.Tokenizer tokenizer(AnalysisOptions.Tokenizer value) {
    return switch (value) {
      case TOKENIZER_UNSPECIFIED, TOKENIZER_WHITESPACE -> PipelineOptions.Tokenizer.WHITESPACE;
      case TOKENIZER_SIMPLE -> PipelineOptions.Tokenizer.SIMPLE;
      case TOKENIZER_UAX29 -> PipelineOptions.Tokenizer.UAX29;
      case TOKENIZER_LATTICE -> PipelineOptions.Tokenizer.LATTICE;
      case TOKENIZER_SENTENCEPIECE -> PipelineOptions.Tokenizer.SENTENCEPIECE;
      case UNRECOGNIZED -> throw Status.INVALID_ARGUMENT
          .withDescription("unrecognized tokenizer value").asRuntimeException();
    };
  }

  /**
   * Proto {@code STEMMER_*} names map onto {@link PipelineOptions.Stemmer} by
   * stripping the prefix; both enums are kept in the same naming convention.
   */
  private static PipelineOptions.Stemmer stemmer(AnalysisOptions.Stemmer value) {
    if (value == AnalysisOptions.Stemmer.UNRECOGNIZED) {
      throw Status.INVALID_ARGUMENT
          .withDescription("unrecognized stemmer value").asRuntimeException();
    }
    if (value == AnalysisOptions.Stemmer.STEMMER_UNSPECIFIED) {
      return PipelineOptions.Stemmer.NONE;
    }
    return PipelineOptions.Stemmer.valueOf(value.name().substring("STEMMER_".length()));
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

  private static PipelineOptions.TermVectorSource termVectorSource(
      TermVectorOptions.Source value) {
    return switch (value) {
      case SOURCE_UNSPECIFIED, SOURCE_TOKENS -> PipelineOptions.TermVectorSource.TOKENS;
      case SOURCE_STEMS -> PipelineOptions.TermVectorSource.STEMS;
      case SOURCE_NORMALIZED_STEMS -> PipelineOptions.TermVectorSource.NORMALIZED_STEMS;
      case UNRECOGNIZED -> throw Status.INVALID_ARGUMENT
          .withDescription("unrecognized term vector source value").asRuntimeException();
    };
  }

  /**
   * Proto {@code NORMALIZER_STEP_*} names map onto
   * {@link PipelineOptions.NormalizerStep} by stripping the prefix.
   * UNSPECIFIED steps are ignored per the contract.
   */
  private static PipelineOptions.NormalizerStep step(
      TermVectorOptions.NormalizerStep value) {
    if (value == TermVectorOptions.NormalizerStep.UNRECOGNIZED) {
      throw Status.INVALID_ARGUMENT
          .withDescription("unrecognized normalizer step value").asRuntimeException();
    }
    if (value == TermVectorOptions.NormalizerStep.NORMALIZER_STEP_UNSPECIFIED) {
      return null;
    }
    return PipelineOptions.NormalizerStep.valueOf(
        value.name().substring("NORMALIZER_STEP_".length()));
  }

  private static PipelineOptions.EmbeddingSource embeddingSource(
      ai.pipestream.opennlp.analysis.v1.EmbeddingOptions.Source value) {
    return switch (value) {
      case SOURCE_UNSPECIFIED, SOURCE_SENTENCES -> PipelineOptions.EmbeddingSource.SENTENCES;
      case SOURCE_TOKENS -> PipelineOptions.EmbeddingSource.TOKENS;
      case UNRECOGNIZED -> throw Status.INVALID_ARGUMENT
          .withDescription("unrecognized embedding source value").asRuntimeException();
    };
  }
}
