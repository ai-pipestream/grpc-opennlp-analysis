# grpc-opennlp-analysis

## Repository map

| Repository | Role | Depends on |
|---|---|---|
| [RyanCodrai/turbovec](https://github.com/RyanCodrai/turbovec) | Upstream vector index library: 4-bit TurboQuant encoding, SIMD top-k search | — |
| [ai-pipestream/turbovec](https://github.com/ai-pipestream/turbovec), branch `turbovec-pipestream` | Patch fork carrying the two core changes distributed search needs: seeded TQ+ calibration and a seedable top-k floor (`initial_threshold`). Rebased onto upstream main | upstream `main` |
| [ai-pipestream/turbovec-grpc](https://github.com/ai-pipestream/turbovec-grpc) | Standalone single-node gRPC server for the upstream index, with client examples in Go, Java, Python, TypeScript, and Rust | upstream `turbovec` |
| [ai-pipestream/turbovec-search](https://github.com/ai-pipestream/turbovec-search) | Distributed hybrid search: sharded vector + BM25 nodes, coordinator with floor sharing, write-ahead log, offline resharding | fork branch `turbovec-pipestream` |
| [ai-pipestream/grpc-opennlp-analysis](https://github.com/ai-pipestream/grpc-opennlp-analysis) (this repo) | Text-analysis sidecar: sentence/token spans, term vectors, static embeddings, served over gRPC | — |

A pure-gRPC NLP analysis service wrapping Apache OpenNLP 3.x, built for
downstream search consumers (turbovec BM25/vector indexing). One `Analyze`
call turns raw text into tokens, sentences, stems, entities, term vectors, and
chunk embeddings — all with **original-text offsets in an explicit unit** — and `AnalyzeStream`
carries the same analysis for bulk indexing: many documents over one bidi
call, paced by server-side flow control.

> **Caveat: experimental dependency.**
> This service runs an **experimental, unmerged Apache OpenNLP 3.x preview
> build**, published as `ai.pipestream:opennlp-*:3.x-preview-SNAPSHOT` from a
> feature fork. It is not an Apache release. The intent is to migrate to the
> official `org.apache.opennlp` 3.x artifacts as the preview features land
> upstream — the swap should be a coordinate change in
> `gradle/libs.versions.toml`, not a code change.

## Why offsets matter

Every `Span` in every response is expressed against the **original request
text** — even for terms produced over normalized text. UTF-16 code units remain
the backward-compatible default; a request may instead select UTF-8 byte
offsets, and every response echoes the resolved unit.
Term identity runs through OpenNLP's offset-aware `Alignment`: with the
`FULL_CASE_FOLD` step, `"Groß"`, `"groß"`, and `"GROSS"` collapse into one term
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
      "steps": ["NORMALIZER_STEP_WHITESPACE", "NORMALIZER_STEP_FULL_CASE_FOLD"]
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
| `tokenizer` | `Tokenizer` | `TOKENIZER_WHITESPACE` | `WHITESPACE` (split on space), `SIMPLE` (letters/digits/punct classes), or `UAX29` (Unicode word segmentation) — all model-free. `LATTICE` (MeCab CJK) and `SENTENCEPIECE` (subword) need server-configured files; unconfigured → whitespace fallback + warning |
| `sentence_detection` | bool | `false` | Newline-based, model-free, exact offsets. Also scopes tokenization |
| `pos_tags` | bool | `false` | Needs a server-configured POS model; otherwise a warning, no tags |
| `ner` | bool | `false` | Needs a server-configured NER model; otherwise a warning, no entities |
| `stemmer` | `Stemmer` | `STEMMER_NONE` | See stemmers below. One stem per token, parallel to `tokens` |
| `lemmatize` | bool | `false` | Needs a server-configured lemmatizer dictionary; otherwise a warning, no lemmas |
| `term_vectors` | `TermVectorOptions` | disabled | `enabled`, `mode` (`MODE_FULL` with occurrence spans / `MODE_SCORING_ONLY` frequency only), `steps`, `source` (`SOURCE_TOKENS` / `SOURCE_STEMS` / `SOURCE_NORMALIZED_STEMS`) |
| `embeddings` | `EmbeddingOptions` | disabled | `source`: `SOURCE_SENTENCES` (chunk embeddings) or `SOURCE_TOKENS` |
| `noise` | bool | `false` | Model-free structural noise scoring (OCR damage, gibberish, base64-ish stretches) |
| `artifacts` | bool | `false` | Model-free text-artifact flagging (replacement chars, mojibake, …). Flags only; never rewrites text |
| `glossary` | `GlossaryOptions` | disabled | Aho-Corasick multi-word phrase matching over request-supplied entries. Model-free |
| `pii` | bool | `false` | Email/phone/IBAN/card extraction with normalized forms. Model-free |
| `coref` | bool | `false` | Deterministic sieve coreference. Needs POS + NER models (implies `pos_tags` + `ner`); otherwise a warning |
| `dependency_parse` | bool | `false` | Needs a dependency-parsing model + POS model (implies `pos_tags`); otherwise a warning |
| `relations` | `RelationOptions` | disabled | Dependency-path patterns between entities. Needs NER + depparse models; otherwise a warning |
| `geo` | bool | `false` | Geocodes location entities against the bundled Natural Earth gazetteer. Needs an NER model (implies `ner`); otherwise a warning |
| `offset_unit` | `OffsetUnit` | `OFFSET_UNIT_UTF16_CODE_UNITS` | Coordinate system for every returned span: Java/OpenNLP UTF-16 code units or original-text UTF-8 bytes |

`AnalyzeResponse`:

- `offset_unit` — resolved coordinate system shared by every returned span;
  an absent value from an older server means UTF-16 code units
- `sentences` — sentence spans in original text coordinates
- `tokens` — span + covered text (+ `pos` when a POS model served the request)
- `stems` — one stem per token, exactly parallel to `tokens`
- `lemmas` — one lemma per token, exactly parallel to `tokens`; unknown words
  come back as `"O"` (the OpenNLP convention)
- `entities` — span + type + covered text
- `term_vectors` — `term`, `frequency`, and (in `MODE_FULL`) every occurrence
  span in **original text coordinates**
- `embeddings` — span + covered text + float vector, parallel to the source
  layer's annotations
- `noise` — span + severity (`misspelled`/`damaged`/`gibberish`/`binaryish`)
  + score in (0, 1]
- `artifacts` — span + type (`replacement`, `control`, `mojibake`, …)
- `glossary_matches` — span + entry id + registered term, leftmost-longest,
  non-overlapping
- `pii` — span + type (`email`/`phone`/`iban`/`card`) + normalized form
- `coref_mentions` — span + chain id + kind (`entity`/`pronoun`) + entity
  index (−1 for pronouns)
- `dependencies` — one arc per token: head/dependent token indices + relation
  label; the root arc has head −1
- `relations` — span + pattern type + subject/object entity indices
- `locations` — geocoded entity: span + gazetteer name + country code +
  coordinates + confidence
- `regions` — document-level country votes (country code + share)
- `warnings` — requested-but-unavailable features (no embedding model loaded,
  no POS/NER model configured, …). Requests never fail just because an
  optional model-backed feature is unavailable.

Changing `offset_unit` changes only the numeric ruler, never tokenization,
normalization, term identity, covered text, or scoring. UTF-8 mode refuses a
pipeline that produces a boundary inside a surrogate pair; use a
Unicode-scalar-safe tokenizer such as `TOKENIZER_UAX29` or
`TOKENIZER_WHITESPACE` for text containing supplementary characters.

Errors: empty text or text over the size cap → `INVALID_ARGUMENT`; term
vectors with `source: SOURCE_STEMS` but no stemmer → `INVALID_ARGUMENT`;
analysis failures → `INTERNAL` with a clean message. No stack traces on the
wire.

### `AnalyzeStream(stream AnalyzeStreamRequest) → stream AnalyzeStreamResponse`

Bulk analysis over one bidi call: the transport for indexing pipelines that
push millions of documents, where per-call dispatch and client-side
concurrency guessing add up. The first message carries `AnalysisOptions` and
fixes the pipeline for the whole stream (two option-sets → two streams;
streams are cheap on HTTP/2). Every later message is a document tagged with a
client-chosen `sequence`, echoed on its response.

The contract, in the order it matters:

- **Responses arrive in completion order, not request order.** The server
  analyzes documents concurrently; match responses by `sequence`. A slow
  document never blocks finished ones behind it.
- **Per-document failures do not fail the stream.** Empty text, oversized
  text, or an analysis failure produce an `AnalyzeStreamResponse` carrying an
  error (canonical gRPC code + message) for that sequence; the stream
  continues. Only protocol violations end the call: a first message that is
  not options, or a second options message → `INVALID_ARGUMENT`.
- **Flow control is server-granted.** The server disables automatic inbound
  requests and grants messages from its actual worker capacity
  (`OPENNLP_STREAM_WORKERS`, default: processor count; window 2× workers), so
  a client that writes as the stream allows is paced by the machine doing the
  work, with backpressure propagating through HTTP/2 to the producer.
- A document that succeeds returns an `ok` payload byte-identical to what
  `Analyze` would return for the same text and options.

### Stemmers

41 algorithmic stemmers plus Hunspell — all model-free:

- **Classic Porter**: `STEMMER_PORTER`.
- **Snowball** (`STEMMER_SNOWBALL_*`): `ENGLISH` (Porter2), `GERMAN`,
  `FRENCH`, `SPANISH`, `ARABIC`, `CATALAN`, `DANISH`, `DUTCH`, `FINNISH`,
  `GREEK`, `HUNGARIAN`, `INDONESIAN`, `IRISH`, `ITALIAN`, `NORWEGIAN`,
  `PORTER` (the Snowball Porter program), `PORTUGUESE`, `ROMANIAN`, `RUSSIAN`,
  `SWEDISH`, `TURKISH`.
- **Light** (`STEMMER_LIGHT_*`): `ENGLISH` (English minimal), `GERMAN`,
  `FRENCH`, `SPANISH`, `FINNISH`, `HUNGARIAN`, `ITALIAN`,
  `NORWEGIAN_BOKMAAL`, `NORWEGIAN_NYNORSK`, `PORTUGUESE`, `RUSSIAN`,
  `SWEDISH`.
- **Minimal** (`STEMMER_MINIMAL_*`): `GERMAN`, `FRENCH`,
  `NORWEGIAN_BOKMAAL`, `NORWEGIAN_NYNORSK`, `SPANISH`, `SWEDISH`.
- **Hunspell**: `STEMMER_HUNSPELL`, driven by a server-configured dictionary
  (`OPENNLP_HUNSPELL_AFF` + `OPENNLP_HUNSPELL_DIC`); unconfigured → warning.

Stemmers operate on the raw token surface form: they do **not** lowercase or
fold (`Running` stems to `Run`, not `run`). Fold client-side or use
token-sourced term vectors with `FULL_CASE_FOLD` when case-insensitive
identity is needed. Stemmers are wrapped in OpenNLP's `SharingStemmer`, so
the shared cached pipelines stay thread-safe.

### Stemmed term vectors (the BM25 feature)

`TermVectorOptions.source` decides what a term **is**:

- `SOURCE_TOKENS` (default): terms are token surface forms after the aligned
  normalizer chain; the steps define identity (`Groß`/`GROSS` collapse under
  `FULL_CASE_FOLD`).
- `SOURCE_STEMS`: **the stem is the identity** — every occurrence of
  `running`/`runs`/`run` groups under `run`, with the occurrence spans of
  every token carrying that stem, in original-text coordinates. The steps are
  ignored in this mode. This is the shape turbovec's BM25 indexing consumes:
  document frequencies and postings over stems, highlightable through the
  original spans. Requires a stemmer (`STEMMER_NONE` → `INVALID_ARGUMENT`).

```bash
grpcurl -plaintext -d '{
  "text": "running runs run",
  "options": {
    "stemmer": "STEMMER_PORTER",
    "termVectors": {"enabled": true, "source": "SOURCE_STEMS", "mode": "MODE_FULL"}
  }
}' localhost:50051 ai.pipestream.opennlp.analysis.v1.AnalysisService/Analyze
# -> "termVectors": [{"term": "run", "frequency": 3,
#      "occurrences": [{0-7}, {8-12}, {13-16}]}]
```

### Lemmatization

`lemmatize: true` joins a server-configured backend onto the tokens, one
lemma per token. Three backends, in priority order: a WordNet database
directory (`OPENNLP_WORDNET_DIR`, the WNdb `data.*`/`index.*`/`*.exc` files
— the Morphy lemmatizer, real validated lemmas), a morfologik morphological
dictionary (`OPENNLP_MORFOLOGIK_DICT`, a CFSA2/FSA5 `.dict` automaton with
its `.info` metadata sibling, as shipped in LanguageTool), or a flat
dictionary (`OPENNLP_LEMMATIZER_DICT`, lines of `word<TAB>postag<TAB>lemma`).
When several are set the highest-priority one wins and each shadowed setting
adds a startup warning. A configured morfologik `.dict` that does not exist
fails startup; a missing `.info` sibling disables the backend with a warning.
Lookup keys on (token, POS
tag); the WordNet backend adapts UD tags (`PROPN` lemmatizes as a noun) and
falls back to an all-POS lookup when no POS model produced tags, while the
dictionary backends look up with the neutral empty tag. Unknown words come
back as `"O"`. Unconfigured → warning, no lemmas.

### Normalizer steps

`steps` on `TermVectorOptions` select the normalizer chain used for term
identity (token-sourced vectors only): `NORMALIZER_STEP_STRIP_INVISIBLE`,
`..._WHITESPACE`, `..._DASHES`, `..._QUOTES`, `..._DIGITS`,
`..._FULL_CASE_FOLD`, `..._ELLIPSIS`, `..._BULLETS`,
`..._EMOJI_TO_EMOTICON`, `..._EMOTICON_TO_EMOJI`, `..._GERMAN_UMLAUT`
(umlaut/sharp-s expansion: `ä`→`ae`, `ß`→`ss`), plus `..._NFKC` / `..._NFC`
(Unicode normalization), `..._CONFUSABLE_SKELETON` (UTS #39 homoglyph
folding), `..._ACCENT_FOLD`, `..._CASE_FOLD` (simple fold; keeps `ß`),
`..._LINE_BREAK_PRESERVING_WHITESPACE`, `..._URL`, `..._NUMBER`,
`..._SOCIAL_MEDIA`, and `..._SHRINK` (repeated-character runs shrink to two),
plus `..._DEHYPHENATE` (line-break hyphenation joining:
`litiga-\ntion`→`litigation`). `DEHYPHENATE` is the one count-changing step:
it is valid only for `SOURCE_TOKENS` identity, where the server re-tokenizes
the normalized text and maps the joined term's span back to the original
offsets (the span covers the hyphen and the line break), and only in an
all-offset-aware chain — combining it with a JDK/regex-backed step or with
`SOURCE_STEMS` fails with `INVALID_ARGUMENT`.
Steps apply in the OpenNLP builder's fixed order regardless of request order.
When `enabled` is set with no steps, the default chain is
`STRIP_INVISIBLE + WHITESPACE + FULL_CASE_FOLD` — the chain BM25 consumers
usually want. Normalization never breaks original-text offsets: steps that
can report a character alignment compose into the aligned whole-document
chain; the JDK/regex-backed steps (NFKC, NFC, confusable skeleton, accent and
case fold, URL, number, social media, shrink) cannot and are applied per
token instead — occurrence spans are the token spans either way.

### `GetCapabilities(GetCapabilitiesRequest) → GetCapabilitiesResponse`

Reports what this server instance can actually serve: OpenNLP and service
versions, `embeddings_enabled` + model dir, POS/NER availability,
`hunspell_available` and `lemmatizer_available` (dictionary configured or
not), `lattice_available`, `sentencepiece_available`, and
`dependency_parse_available` (tokenizer/parser files configured or not), the
max text size, and the supported stemmer/step/tokenizer option names. Noise
scoring, artifact flagging, glossary matching, and PII extraction are
model-free and always available, so they carry no availability flags. Dual
term identity, spell correction, and de-hyphenation additionally report
discovery flags (`dual_term_identity_available`, `spellcheck_available`,
`dehyphenation_available`) so a client compiled against an older contract
can tell the features exist on this server.

## Model-free by default

The default pipeline needs **zero downloads and zero model files**:
whitespace/simple/UAX29 tokenizers, newline sentence detection, algorithmic
stemmers, term vectors, noise scoring, artifact flagging, glossary matching,
and PII extraction. Model-backed features are opt-in via server
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
- **Hunspell stemming** — set `OPENNLP_HUNSPELL_AFF` and
  `OPENNLP_HUNSPELL_DIC` to a Hunspell dictionary pair, then select
  `STEMMER_HUNSPELL` per request.
- **Lemmatization** — set `OPENNLP_WORDNET_DIR` to a WordNet database
  directory (preferred; the Morphy lemmatizer), `OPENNLP_MORFOLOGIK_DICT` to
  a morfologik `.dict` automaton with its `.info` sibling (e.g. the
  LanguageTool English dictionary), or `OPENNLP_LEMMATIZER_DICT`
  to a `word<TAB>postag<TAB>lemma` file, then set `lemmatize: true` per
  request.
- **Spell correction** — set `OPENNLP_SPELLCHECK_MODEL` to a serialized
  SymSpell model, then add `NORMALIZER_STEP_SPELLCHECK` to a term vector
  spec's steps. Model-gated: without a model the step is skipped with a
  warning. Correction shapes term identity only — use it in a parallel
  column's spec, never silently in the body's.
- **Lattice (CJK) tokenizer** — set `OPENNLP_LATTICE_DIC_DIR` to a MeCab
  dictionary directory (IPADIC, UniDic, …), then select `TOKENIZER_LATTICE`.
  Unconfigured requests fall back to whitespace tokenization with a warning.
- **SentencePiece tokenizer** — set `OPENNLP_SENTENCEPIECE_MODEL` to a trained
  `.model` file, then select `TOKENIZER_SENTENCEPIECE`. Same fallback rule.
- **Dependency parsing** — set `OPENNLP_DEPPARSE_MODEL` to a feedforward
  dependency model file, then set `dependency_parse: true`. Implies
  `pos_tags`; relations build on it.
- **Coreference** — `coref: true` needs POS + NER models (deterministic
  sieves; no coref model). Implies `pos_tags` + `ner`.
- **Geocoding** — `geo: true` needs an NER model; location entities resolve
  against the bundled Natural Earth populated-places gazetteer (no external
  data files).

### Fetching dictionaries and models

`ResourceDownloader` is the operator CLI for acquiring optional resources
(nothing is ever downloaded at request time). It builds on the preview's
`ResourceInstaller`: SHA-256 verification, tar.gz/zip unpacking, and
archive-path-escape rejection.

```bash
./gradlew downloadResource --args="list"                                    # the catalog
./gradlew downloadResource --args="install hunspell-en-us /path/to/dir"     # catalog entry
./gradlew downloadResource --args="install <uri> <dir> [sha256-hex]"        # any file/archive
```

Installing a catalog entry prints the exact `OPENNLP_*` exports to hand the
server. The catalog currently covers the LibreOffice Hunspell English (US)
dictionary, the OpenNLP POS/NER models, the Princeton WordNet database, and
the LanguageTool English morfologik dictionary; anything else (a MeCab
dictionary tarball, a SentencePiece
`.model`, a feedforward dependency model, a POS/NER `.bin`) installs by URL.
POS/NER model loading relies on the build's generated `opennlp.version`
resource (see `build.gradle`): the fork's own version string does not parse,
and the generated parseable rendition shadows it on the classpath.

## Configuration

| Setting | Env var | System property | Default |
|---|---|---|---|
| Listen port | `PORT` | `port` (or first CLI arg) | `50051` |
| Max request text bytes | `OPENNLP_MAX_TEXT_BYTES` | `opennlp.max.text.bytes` | `1048576` (1 MiB) |
| Embedding model dir | `OPENNLP_EMBEDDINGS_DIR` | `opennlp.embeddings.dir` | unset (disabled) |
| POS model file | `OPENNLP_POS_MODEL` | `opennlp.pos.model` | unset (unavailable) |
| NER model file | `OPENNLP_NER_MODEL` | `opennlp.ner.model` | unset (unavailable) |
| Hunspell .aff file | `OPENNLP_HUNSPELL_AFF` | `opennlp.hunspell.aff` | unset (unavailable) |
| Hunspell .dic file | `OPENNLP_HUNSPELL_DIC` | `opennlp.hunspell.dic` | unset (unavailable) |
| WordNet database dir | `OPENNLP_WORDNET_DIR` | `opennlp.wordnet.dir` | unset (unavailable) |
| Morfologik dictionary | `OPENNLP_MORFOLOGIK_DICT` | `opennlp.morfologik.dict` | unset (unavailable) |
| Lemmatizer dictionary | `OPENNLP_LEMMATIZER_DICT` | `opennlp.lemmatizer.dict` | unset (unavailable) |
| Spell-check model file | `OPENNLP_SPELLCHECK_MODEL` | `opennlp.spellcheck.model` | unset (SPELLCHECK step skipped with a warning) |
| MeCab dictionary dir | `OPENNLP_LATTICE_DIC_DIR` | `opennlp.lattice.dic.dir` | unset (LATTICE falls back to whitespace) |
| SentencePiece model file | `OPENNLP_SENTENCEPIECE_MODEL` | `opennlp.sentencepiece.model` | unset (SENTENCEPIECE falls back to whitespace) |
| Dependency model file | `OPENNLP_DEPPARSE_MODEL` | `opennlp.depparse.model` | unset (unavailable) |
| AnalyzeStream workers | `OPENNLP_STREAM_WORKERS` | `opennlp.stream.workers` | `0` (processor count) |

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
