# Vocabulary listener — design

Status: design agreed 2026-08-05, implementation in progress on this branch.

## The problem it solves

We want the search engine's embedding columns to be produced by static
(Model2Vec-family) models that are *continuously re-tuned against the corpus
they index*, with new model versions producing candidate indexes that live
A/B testing promotes or discards. That loop needs two instruments this
service does not yet have:

1. A **vocabulary index**: streaming statistics over every term the
   analytics pipeline produces during indexing, harvested for free while the
   corpus is being indexed anyway.
2. **Drift analytics**: quantified answers to "how much has the corpus's
   vocabulary moved since the current embedding model's training data was
   collected?" — the signal that decides *when* a reweight is warranted.
   A/B testing remains the arbiter of *whether* a candidate promotes.

Corpora always drift; that is the premise of the continuous loop, not a
problem to fence off. The listener measures the drift; it does not gate on
it.

## Where it lives — final architecture

The listener is split across the two services by role:

- **Java (this sidecar) calculates.** It is the analytics layer: it sees
  the pipeline's output, so term identity is counted exactly as produced —
  the vocabulary index and the BM25 index can never disagree about what "a
  term" is. AnalyzeStream is the bulk/indexing path; unary `Analyze` is the
  query path, and query text never enters corpus statistics.
- **Rust (turbovec-search) accumulates, stores, and aggregates.** And the
  key realization: **no new gRPC channel is needed for this.** Rust's
  ingest path already receives every document's tokens and term vectors in
  the AnalyzeStream responses it consumes to build postings — the "push
  stream" already exists. Rust updates the sketches inline while indexing,
  so windows naturally align with index generations, snapshots persist in
  the index's own territory, and aggregation across shards is a
  coordinator-side sketch merge (HLL and count-min merge
  losslessly-in-expectation; heavy hitters merge by re-running space-saving
  over the combined lists). The vocabulary index is a property of the
  indexed corpus; its system of record is the engine.

The Java `VocabularyService` in this repo remains as the **generic tap**:
the accumulation point for corpus sources that do not flow through
turbovec-search ingest (e.g. a future pipestream-side pipeline). It is
disabled unless `OPENNLP_VOCAB_DIR` is set, costs nothing when off, and its
snapshot format is the shared contract both sides speak. A dedicated
push/pull channel between the services is deliberately NOT built: every
current consumer is covered by the two taps above, and the proto messages
are already there if a genuine second consumer ever needs streaming
delivery.

## What it counts: two channels

Every analyzed document feeds two independent channels:

| Channel | Source | Identity | Purpose |
|---|---|---|---|
| TERMS | `term_vectors` of the response (folded source) | exactly the BM25 index identity | BM25 drift, term-frequency shift, heavy hitters |
| TOKENS | raw token text of the response | surface forms, pre-normalization | embedding-vocab coverage (Model2Vec reweight input), OOV estimation |

A document contributes its term-vector entries with their frequencies
(TERMS) and its raw token strings (TOKENS). Documents whose request did not
ask for term vectors still feed TOKENS. Both channels share the windowing
and snapshot machinery below.

## Data structures (per channel, per window)

Memory is bounded by configuration, never by corpus size:

- **HyperLogLog** (64-bit hashes, p=14 → 16 KiB registers): distinct-term
  cardinality. Standard error ~0.8%. Mergeable, which gives novelty via
  inclusion-exclusion.
- **Count-min sketch** (depth 5, width 2^17 → 5 MiB of longs): per-term
  frequency estimates, point queries for terms missing from the heavy-hitter
  list.
- **Space-saving top-K** (K=1024): the heaviest terms with near-exact
  counts. This is what JS divergence is computed over, and what a
  Model2Vec reweight actually consumes (the tail beyond top-K contributes
  negligible mass).
- **Counters**: documents, total term occurrences.

At 86.6M chunks a window holds ~11 MiB per channel; two channels plus a
handful of retained snapshots is tens of MiB — irrelevant against the
models this service already loads.

## Windowing and snapshots

- A **window** accumulates until it rolls over, at which point it is sealed
  and persisted as a **snapshot**. Rollover happens three ways: the window
  reaches `OPENNLP_VOCAB_WINDOW_DOCS` documents (default 1,000,000), an
  explicit `SnapshotVocab` rpc, or JVM shutdown (best effort).
- Snapshots are protobuf messages (`VocabSnapshot`) written to
  `OPENNLP_VOCAB_DIR/snapshot-<seq>-<epochMillis>.pb`. Protobuf rather than
  Java serialization: snapshots are training data for the embedding
  reweight, so they must be readable by non-JVM tooling.
- At startup the listener scans the snapshot directory and indexes prior
  snapshots (metadata only; full sketches are loaded lazily when a drift
  comparison names them). A reindex therefore *resumes* its vocabulary
  history instead of starting over.
- The HLL registers, count-min table, and heavy-hitter list all serialize
  into the snapshot, so every metric below remains computable between any
  two historical windows, not just adjacent ones.

## Drift metrics

`GetVocabDrift(from, to)` compares two snapshots (or the live window
against a snapshot) per channel and reports:

- **Cardinality**: distinct-term estimates for both windows and their union
  (merged HLL).
- **Novelty rate**: `(|A ∪ B| − |A|) / |B|` — the share of the newer
  window's distinct vocabulary that the older window never saw. This is the
  primary "the corpus moved" number.
- **Frequency shift**: Jensen-Shannon divergence over the union of the two
  top-K lists, with missing counts filled from the count-min sketches.
  Distribution-level movement even when the vocabulary itself is stable.
- **Embedding coverage** (TOKENS channel, when an embedding model is
  configured): the share of the newer window's *token mass* whose surface
  form is absent from the live embedding model's vocabulary (read from the
  model directory's `vocab.txt` / `tokenizer.json`, not from a private
  API). When this climbs, the deployed static model is degrading on fresh
  text even if old queries still look fine — the reweight trigger.

None of these gate anything automatically. They inform the operator's
decision to train a candidate; promotion is always decided by live A/B.

## gRPC contract

A separate `VocabularyService` in the same proto file (same server port,
registered only when the listener is enabled):

```proto
service VocabularyService {
  rpc GetVocabStats(GetVocabStatsRequest) returns (GetVocabStatsResponse);
  rpc SnapshotVocab(SnapshotVocabRequest) returns (SnapshotVocabResponse);
  rpc GetVocabDrift(GetVocabDriftRequest) returns (GetVocabDriftResponse);
}
```

- `GetVocabStats`: listener enabled state, live-window counters and
  cardinality estimates per channel, and the list of persisted snapshots
  (sequence, timestamps, doc counts).
- `SnapshotVocab`: seals and persists the live window, returns its
  snapshot descriptor.
- `GetVocabDrift`: the metrics above between two named snapshots (either
  side may be `"live"` for the current window).

`GetCapabilities` gains `vocab_listener_available` (field 20), per the
service's standing rule that every new capability is discoverable so a
client can tell an old sidecar from a new one.

The vendored copy at
`turbovec-search/proto/ai/pipestream/opennlp/analysis/v1/analysis.proto`
must be updated byte-identically in the same change (their build
diff-checks it).

## Configuration

| Setting | Property | Default | Meaning |
|---|---|---|---|
| `OPENNLP_VOCAB_DIR` | `opennlp.vocab.dir` | unset | Snapshot directory. **Unset = listener disabled**, zero overhead, service not registered. |
| `OPENNLP_VOCAB_WINDOW_DOCS` | `opennlp.vocab.window.docs` | 1000000 | Documents per window before automatic rollover. |
| `OPENNLP_VOCAB_TOP_K` | `opennlp.vocab.top.k` | 1024 | Heavy-hitter list size. |

## Failure modes, chosen deliberately

- **Listener throws during ingest**: counted terms are lost for that
  document; the document's analysis response is unaffected. The listener
  must never fail an ingest. (Implementation: the feed call is wrapped and
  logged, never propagated.)
- **Process killed before rollover**: the live window is lost; snapshots
  on disk are intact. Vocabulary statistics are approximate by design, so a
  lost partial window is acceptable — this is analytics, not a ledger.
- **Snapshot directory unwritable**: listener starts disabled with a loud
  warning in `GetCapabilities.warnings`, analysis unaffected.
- **Two sidecar instances** (as on this fleet: :59202 and :59220) each keep
  their own vocabulary history under their own `OPENNLP_VOCAB_DIR`.
  Merging across instances is a tooling problem (sketches merge cleanly),
  not a service concern.

## Explicit non-goals (for this change)

- **No model training.** Snapshots are the *input* to the Model2Vec
  reweight; the training job is a separate subsystem that reads them.
- **No gating or automation.** Nothing here triggers a reweight, builds an
  index, or swaps anything. The A/B harness owns promotion.
- **No query-side listener.** Query vocabulary drift is a different
  (also interesting) signal; it is not needed for the training loop and is
  deliberately out of scope.
- **No cross-instance federation.** One listener per sidecar process.

## Deployment notes

- The live fleet runs the GraalVM native binary. JVM-side verification
  (`./gradlew assemble test`) comes first; the native rebuild and any
  service restart are a separate, deliberate act — per the ground rules,
  `installDist` + restart of the live sidecar is never casual.
- Enabling on the fleet means setting `OPENNLP_VOCAB_DIR` per instance and
  restarting — which pairs naturally with the next planned reindex so the
  pass harvests v2 training data for free.
