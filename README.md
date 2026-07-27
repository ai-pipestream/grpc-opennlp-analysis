# grpc-opennlp-analysis

A pure-gRPC NLP analysis service wrapping Apache OpenNLP 3.x, built for
downstream search consumers (turbovec BM25/vector indexing). One `Analyze`
call turns raw text into tokens, sentences, stems, entities, term vectors, and
chunk embeddings — all with **original-text offsets**.

> **Caveat: experimental dependency.**
> This service runs an **experimental, unmerged Apache OpenNLP 3.x preview
> build**, published as `ai.pipestream:opennlp-*:3.x-preview-SNAPSHOT` from a
> feature fork. It is not an Apache release. The intent is to migrate to the
> official `org.apache.opennlp` 3.x artifacts as the preview features land
> upstream — the swap should be a coordinate change in
> `gradle/libs.versions.toml`, not a code change.

## Why offsets matter

Every `Span` in every response is expressed in the character offsets of the
**original request text** — even for terms produced over normalized text.
Term identity runs through OpenNLP's offset-aware `Alignment`: with the
`FULL_CASE_FOLD` rung, `"Groß"`, `"groß"`, and `"GROSS"` collapse into one term
`gross`, while each occurrence still carries its exact original span
(`[0,4)`, `[6,10)`, `[12,17)`). This is what makes **true highlighting**
possible for both BM25 and vector search results: you can always paint the
exact characters the user typed, never a re-normalized approximation.

## Quickstart

Requires JDK 25+.

```bash
./gradlew build          # compile, codegen, test
./gradlew run            # starts on port 50051 (or PORT / first arg)
```

Exercise it with [grpcurl](https://github.com/fullstorydev/grpcurl)
(server reflection is enabled, no proto files needed):

```bash
# Capabilities
grpcurl -plaintext localhost:50051 \
  ai.pipestream.opennlp.analysis.v1.AnalysisService/GetCapabilities

# Analyze: tokens, sentences, porter stems, full term vectors
grpcurl -plaintext -d '{
  "text": "Groß  groß  GROSS",
  "options": {
    "sentenceDetection": true,
    "stemmer": "STEMMER_PORTER",
    "termVectors": {
      "enabled": true,
      "mode": "MODE_FULL",
      "rungs": ["NORMALIZER_RUNG_WHITESPACE", "NORMALIZER_RUNG_FULL_CASE_FOLD"]
    }
  }
}' localhost:50051 ai.pipestream.opennlp.analysis.v1.AnalysisService/Analyze
```

The term vector section of the response:

```json
"termVectors": [{
  "term": "gross",
  "frequency": 3,
  "occurrences": [
    {"start": 0,  "end": 4},
    {"start": 6,  "end": 10},
    {"start": 12, "end": 17}
  ]
}]
```

## RPC surface

Package `ai.pipestream.opennlp.analysis.v1`, file
[`src/main/proto/ai/pipestream/opennlp/analysis/v1/analysis.proto`](src/main/proto/ai/pipestream/opennlp/analysis/v1/analysis.proto)
(`buf lint` STANDARD clean).

### `Analyze(AnalyzeRequest) → AnalyzeResponse`

`AnalyzeRequest` is `text` plus `AnalysisOptions`:

| Option | Type | Default | Notes |
|---|---|---|---|
| `language` | string | `"en"` | Informational in the model-free pipeline; selects configured model language |
| `tokenizer` | `Tokenizer` | `TOKENIZER_WHITESPACE` | `WHITESPACE` (split on space) or `SIMPLE` (letters/digits/punct classes). Model-free |
| `sentence_detection` | bool | `false` | Newline-based, model-free, exact offsets. Also scopes tokenization |
| `pos_tags` | bool | `false` | Needs a server-configured POS model; otherwise a warning, no tags |
| `ner` | bool | `false` | Needs a server-configured NER model; otherwise a warning, no entities |
| `stemmer` | `Stemmer` | `STEMMER_NONE` | See stemmers below. One stem per token, parallel to `tokens` |
| `term_vectors` | `TermVectorOptions` | disabled | `enabled`, `mode` (`MODE_FULL` with occurrence spans / `MODE_SCORING_ONLY` frequency only), `rungs` |
| `embeddings` | `EmbeddingOptions` | disabled | `source`: `SOURCE_SENTENCES` (chunk embeddings) or `SOURCE_TOKENS` |

`AnalyzeResponse`:

- `sentences` — sentence spans in original text coordinates
- `tokens` — span + covered text (+ `pos` when a POS model served the request)
- `stems` — one stem per token, exactly parallel to `tokens`
- `entities` — span + type + covered text
- `term_vectors` — `term`, `frequency`, and (in `MODE_FULL`) every occurrence
  span in **original text coordinates**
- `embeddings` — span + covered text + float vector, parallel to the source
  layer's annotations
- `warnings` — requested-but-unavailable features (no embedding model loaded,
  no POS/NER model configured, …). Requests never fail just because an
  optional model-backed feature is unavailable.

Errors: empty text or text over the size cap → `INVALID_ARGUMENT`; analysis
failures → `INTERNAL` with a clean message. No stack traces on the wire.

### Stemmers

All model-free and algorithmic: `STEMMER_PORTER`, `STEMMER_SNOWBALL_ENGLISH`,
`STEMMER_SNOWBALL_GERMAN`, `STEMMER_SNOWBALL_FRENCH`,
`STEMMER_SNOWBALL_SPANISH`, `STEMMER_LIGHT_ENGLISH`, `STEMMER_LIGHT_GERMAN`,
`STEMMER_LIGHT_FRENCH`, `STEMMER_LIGHT_SPANISH`. Stemmers are wrapped in
OpenNLP's `SharingStemmer`, so the shared cached pipelines stay thread-safe.

### Normalizer rungs

`rungs` on `TermVectorOptions` select the aligned normalizer chain used for
term identity: `NORMALIZER_RUNG_STRIP_INVISIBLE`, `..._WHITESPACE`,
`..._DASHES`, `..._QUOTES`, `..._DIGITS`, `..._FULL_CASE_FOLD`. Rungs apply in
the OpenNLP builder's fixed order regardless of request order. When `enabled`
is set with no rungs, the default chain is
`STRIP_INVISIBLE + WHITESPACE + FULL_CASE_FOLD` — the chain BM25 consumers
usually want. Every rung is offset-aware: normalization never breaks
original-text offsets.

### `GetCapabilities(GetCapabilitiesRequest) → GetCapabilitiesResponse`

Reports what this server instance can actually serve: OpenNLP and service
versions, `embeddings_enabled` + model dir, POS/NER availability, the max text
size, and the supported stemmer/rung/tokenizer option names.

## Model-free by default

The default pipeline needs **zero downloads and zero model files**:
whitespace/simple tokenizers, newline sentence detection, algorithmic
stemmers, aligned term vectors. Model-backed features are opt-in via server
configuration and are never downloaded at request time:

- **Embeddings** — set `OPENNLP_EMBEDDINGS_DIR` to a static embedding model
  directory. The backend is OpenNLP's pure-Java `StaticEmbeddingModel`, which
  auto-detects the **Model2Vec** layout (`model.safetensors` +
  `tokenizer.json` + `config.json`, as produced by
  [`minishlab` Model2Vec](https://github.com/MinishLab/model2vec) exports).
  When unset, the service runs with embeddings disabled: embedding requests
  succeed with a warning and no vectors, and `GetCapabilities` reports
  `embeddings_enabled: false`.
- **POS tagging** — set `OPENNLP_POS_MODEL` to a `.bin` POS model file.
- **NER** — set `OPENNLP_NER_MODEL` to a `.bin` token name finder model file.
  The name finder is serialized behind a lock (it clears adaptive data per
  call), so shared pipelines stay safe.

## Configuration

| Setting | Env var | System property | Default |
|---|---|---|---|
| Listen port | `PORT` | `port` (or first CLI arg) | `50051` |
| Max request text bytes | `OPENNLP_MAX_TEXT_BYTES` | `opennlp.max.text.bytes` | `1048576` (1 MiB) |
| Embedding model dir | `OPENNLP_EMBEDDINGS_DIR` | `opennlp.embeddings.dir` | unset (disabled) |
| POS model file | `OPENNLP_POS_MODEL` | `opennlp.pos.model` | unset (unavailable) |
| NER model file | `OPENNLP_NER_MODEL` | `opennlp.ner.model` | unset (unavailable) |

The server exposes the gRPC **health service** (`""` and the fully-qualified
service name are `SERVING`) and **server reflection**. Shutdown is graceful:
`shutdown()`, up to 5 s for in-flight calls, then `shutdownNow()`.

Logging is `slf4j-api` with the **`slf4j-nop`** backend — deliberate: nop is
the native-image-safe choice (netty's build-time class initialization would
otherwise capture a `SimpleLogger` into the GraalVM image heap).

## Development

```bash
./gradlew build       # compile + test
./gradlew test        # tests only (JUnit + AssertJ, JUnit Platform)
buf lint              # proto lint (STANDARD)
buf format -w         # proto formatting
./gradlew bufLint     # the same, via Gradle
```

Layout: single Gradle module. `pipeline/` holds the gRPC-free,
unit-testable pipeline construction (`AnalysisPipeline`, `PipelineOptions`,
`PipelineEnvironment`); the gRPC layer (`AnalysisServiceImpl`, `OptionsMapper`,
`AnalysisServer`) only validates, caches one shared pipeline per distinct
option-set, and maps `Document` layers onto the proto. Tests run both
in-process (`InProcessServerBuilder`) and against a real Netty server on a
random loopback port.

## Native image (GraalVM CE)

The service builds as a GraalVM native image — sub-25 ms startup, no JRE on
the target machine. Building the native binary is the only task that requires
a GraalVM JDK 25 (e.g. `sdk install java 25.1.3-graalce`); every other build
and test task runs on any JDK 25 via Gradle toolchains.

```bash
# Pin the toolchain to the GraalVM install: the toolchain spec is deliberately
# vendor-neutral ("25"), and a temurin 25 cannot run native-image. Restricting
# detection to the graalce installation makes the pick deterministic.
JAVA_HOME=$HOME/.sdkman/candidates/java/25.1.3-graalce ./gradlew nativeCompile \
  -Porg.gradle.java.installations.paths=$HOME/.sdkman/candidates/java/25.1.3-graalce

# Binary:
./build/native/nativeCompile/grpc-opennlp-analysis   # PORT=50051 or first arg
```

Measured on the development machine (GraalVM CE 25.1.3, native-image finishes
in ~18 s):

| Characteristic | Value |
|---|---|
| Binary size | 49.8 MiB (single static executable) |
| Cold start to port-accept | ~22 ms |
| RSS at startup | ~33 MB |
| RSS after 300 RPCs | ~124 MB (netty pooled buffers grow under load) |

Verified natively (via grpcurl against the running binary): server reflection
(`grpcurl list`), `GetCapabilities`, the full `Analyze` surface (sentence
detection, both tokenizers, porter/snowball/light stemmers, FULL and
SCORING_ONLY term vectors with the aligned normalizer chain — the
`"Groß  groß  GROSS"` case returns one term `gross` with occurrences at
`[0,4) [6,10) [12,17)`), unavailable-feature warnings, `INVALID_ARGUMENT`
validation, and the health service.

Native-specific configuration, all in `build.gradle` + one metadata file:

- `org.graalvm.buildtools.native` 1.1.6 with `--no-fallback`,
  `--report-unsupported-elements-at-runtime`, `-H:+ReportExceptionStackTraces`.
  The plugin's metadata repository covers grpc-netty-shaded and protobuf-java.
- `src/main/resources/META-INF/native-image/ai.pipestream/grpc-opennlp-analysis/reachability-metadata.json`
  adds: `ExtensionRegistry.add(Extension)` reflection (protobuf descriptor
  init), `**/*.proto` resources, and the two properties files the service
  reads from the classpath (`service-version.properties`, opennlp-api's
  `pom.properties` for the reported version).
- `slf4j-nop` (not slf4j-simple): netty's build-time class initialization
  would otherwise capture a `SimpleLogger` into the image heap.

Honest limitations: embeddings/POS/NER were not exercised natively (no model
was configured during verification; the code paths are plain Java but
unproven in the image), and native-image emits 21 warnings (mostly netty's
`sun.misc.Unsafe` usage — harmless today, worth watching as JDKs evolve).

## Container

The multi-stage [Dockerfile](Dockerfile) builds the native binary inside a
GraalVM CE 25 image and packs it onto `debian:trixie-slim` — no JRE in the
final image, non-root user (uid 10001), port 50051 exposed.

```bash
docker build -t grpc-opennlp-analysis .
docker run --rm -p 50051:50051 grpc-opennlp-analysis

# With embeddings (mount the Model2Vec directory):
docker run --rm -p 50051:50051 \
  -e OPENNLP_EMBEDDINGS_DIR=/models/model \
  -v /path/to/model:/models/model:ro \
  grpc-opennlp-analysis
```

All configuration env vars (`PORT`, `OPENNLP_EMBEDDINGS_DIR`,
`OPENNLP_POS_MODEL`, `OPENNLP_NER_MODEL`, `OPENNLP_MAX_TEXT_BYTES`) work the
same inside the container. Model paths must refer to mounted volumes.

The final image is ~195 MB (debian base plus the 50 MB binary; the GraalVM
builder stage is discarded). Verified locally: the container answers
`GetCapabilities` and the aligned term-vector `Analyze` case.

## Roadmap

- ~~Phase 2a — GraalVM CE native image~~: done, see above.
- ~~Phase 2b — publishing and CI~~: snapshot jars publish to Central Snapshots
  (github.com) and the ai.pipestream Forgejo registry (self-hosted) via
  Gradle; the native binary is a CI artifact; the Dockerfile is local/manual
  for now (a registry push can come later). The primary consumer is
  **turbovec distributed search**.
- Migration from `ai.pipestream:opennlp-*` snapshots to official
  `org.apache.opennlp` 3.x artifacts as the preview features land upstream.
- Optional statistical sentence detection / tokenizer models via explicit
  model-path configuration.

## License

Apache License 2.0 — see [LICENSE](LICENSE).
