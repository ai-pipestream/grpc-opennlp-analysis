# AGENTS.md

## Remotes and push policy

- **Push Forgejo first, GitHub second.** Forgejo
  (`git.rokkon.com/ai-pipestream/grpc-opennlp-analysis`, remote `forgejo`)
  is the master build; GitHub (remote `origin`) is the public copy. Nothing
  auto-syncs between them — push both, in that order. (GitHub currently
  lags forgejo by a few commits; that is expected, not divergence.)
- Workspace-wide policy and the per-repo remote table live in the
  workspace-root `../AGENTS.md` — read it before pushing anywhere.
