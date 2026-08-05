# AGENTS.md

## Remotes and push policy

- **Push Forgejo first, GitHub second.** Forgejo
  (`git.rokkon.com/ai-pipestream/grpc-opennlp-analysis`, remote `forgejo`)
  is the master build; GitHub (remote `origin`) is the public copy. Nothing
  auto-syncs between them — push both, in that order. (GitHub currently
  lags forgejo by a few commits; that is expected, not divergence.)
- Workspace-wide policy and the per-repo remote table live in the
  workspace-root `../AGENTS.md` — read it before pushing anywhere.

## Vocabulary listener

- Design and rationale: `VOCABULARY-LISTENER.md` (authoritative spec).
  Implementation: `src/main/java/ai/pipestream/opennlp/analysis/vocab/`
  (sketches, window state, listener) plus `VocabServiceImpl`; fed only from
  the AnalyzeStream ingest path in `AnalysisServiceImpl`, never from unary
  Analyze.
- Env vars (sysprop form in parentheses): `OPENNLP_VOCAB_DIR`
  (`opennlp.vocab.dir`) — unset means disabled, zero overhead,
  VocabularyService not registered; `OPENNLP_VOCAB_WINDOW_DOCS`
  (`opennlp.vocab.window.docs`, default 1000000);
  `OPENNLP_VOCAB_TOP_K` (`opennlp.vocab.top.k`, default 1024).
- An unwritable `OPENNLP_VOCAB_DIR` degrades to disabled with a
  GetCapabilities warning; the flag is `vocab_listener_available` (proto
  field 20). The vendored proto copy at
  `turbovec-search/proto/ai/pipestream/opennlp/analysis/v1/analysis.proto`
  must stay byte-identical (their build diff-checks it).
