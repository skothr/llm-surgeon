# HARD RULE: every session MUST work in a git worktree

**Not a guideline. Not a preference. A hard rule.** When two sessions
share the main checkout, uncommitted work from one session gets
accidentally swept into the other's `git add` and commit. Worktrees
are the fix.

## Pre-flight check — BEFORE your first edit/write/bash-write

Run this as the very first thing in any session at this project:

```bash
GIT_DIR=$(cd "$(git rev-parse --git-dir)" && pwd -P)
GIT_COMMON=$(cd "$(git rev-parse --git-common-dir)" && pwd -P)
[ "$GIT_DIR" = "$GIT_COMMON" ] && echo "MAIN CHECKOUT — MUST CREATE WORKTREE" || echo "in worktree — proceed"
```

If the check says "MAIN CHECKOUT": **STOP. Do not edit any file.**
Create a worktree at `.claude/worktrees/<scope>/` on its own branch
(via `EnterWorktree`) before doing anything else. `.claude/` is
gitignored — no .gitignore changes needed.

The only commits permitted directly on the default branch (`main`)
in the main checkout are **integration commits**: merge commits
(`gh pr merge`), or convention-establishing changes to CLAUDE.md and
`.gitignore` itself. Everything else — even one-line typo fixes —
goes through a worktree.

## Session lifecycle

1. **`EnterWorktree name=<scope>`** — creates `.claude/worktrees/<scope>/`
   and switches the session into it. Conventional branch prefixes:
   `feat/`, `fix/`, `refactor/`, `docs/`, `session/`.
2. All edits and commits land on that branch in its worktree. Never edit
   files in another session's sibling worktree (inspect read-only with
   `git -C .claude/worktrees/other log` if needed).
3. Push (`git push -u origin <branch>`) and open a PR (`gh pr create`).
   Merge to `main` via PR — never edit the main checkout directly.
4. After merge, remove the worktree: `git worktree remove
   .claude/worktrees/<scope>` (or `ExitWorktree`).

---

# Purpose

`llm-surgeon` is a standalone, reusable Python library for surgical
layer-level manipulation and interpretability probing of local
LLaMA-family models (HuggingFace + GGUF). The package is `llm_surgeon/`;
tests in `tests/`; generic runnable demos in `examples/`; the phased
design record in `docs/design-history/` (`specs/` + `plans/`).

# TESTING

```bash
# Run the pytest suite from the repo root
pytest
```

- `pyproject.toml` sets `testpaths = ["tests"]` and `pythonpath = ["."]`.
- Install dev deps first: `pip install -e ".[dev]"` (system python is
  not assumed to have torch/pytest).
- `llm_surgeon` is installed editable via `pip install -e .`.

## Dev models

- `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (22 layers, 2048 hidden, 1.1B
  params) — **default** for examples, fixtures, and fast test iteration.
- `openlm-research/open_llama_3b_v2` (26 layers, 3200 hidden, 3B params)
  — for a slightly-larger-scale sanity check.
- Models live in the HuggingFace `models--{org}--{name}` cache layout.

# Type Checking

Project stance: zero errors, warnings, AND informations after every edit
for pyright. `pyrightconfig.json` (extends the shared base config) is set
so the `<new-diagnostics>` linter messages line up with what we want
fixed; running pyright via Bash separately costs time — don't, unless the
user asks or you suspect a cache mismatch.

```bash
# Pyright manual run — only if the user asks, or you suspect a cache mismatch:
.venv/bin/python -m pyright <paths>
```

## Fix patterns

- **Real typing bug** → fix the source. Prefer upstream type tightening
  so narrowing cascades.
- **Ad-hoc dynamic class** (`class _Meta: pass; cfg = _Meta(); cfg.x = ...`)
  → replace with `types.SimpleNamespace(x=..., ...)` — stdlib, typed as
  dynamic.
- **Test helper for `Optional[X]` guarded by `@pytest.mark.skipif`** →
  extract a helper like `_tinyllama_blob() -> Path` that asserts and
  returns the narrowed type.
- **Never** disable rules in `pyrightconfig.json` to quiet diagnostics —
  it hides real bugs elsewhere.

### Type-narrowing tier list

`assert isinstance` > `cast` > `# pyright: ignore[reportXxx]`; never bare
`# type: ignore`. The base config has `reportUnnecessaryTypeIgnoreComment`
ON, so stale `# pyright: ignore` annotations self-surface for deletion
when stubs catch up.

## Known stub lag

- Fully unstubbed packages: `llama_cpp`, `gguf`, `bitsandbytes`.
- Torch stubs lag runtime for: `torch.OutOfMemoryError`,
  `with torch.device(...)`, `load_state_dict(assign=...)`.
- `reportPrivateImportUsage` is muted in the shared base config because
  torch's `__init__.pyi` doesn't re-export the bulk of its runtime
  surface (`torch.float32`, `torch.zeros`, `torch.tensor`, ...).

## Unused symbols — underscore-prefix behavior

**Honored** (rename to `_name` suppresses): local assignments, tuple
unpacking, `for _idx, val in enumerate(...)`, function parameters.

**`reportUnusedFunction` and `reportUnusedClass` are muted in the base
config** — research code legitimately has scratch helpers; the escape
hatches are anti-patterns. Use grep / IDE for real dead-code sweeps.

**`reportUnusedImport` is honored** (rename to `_name` does NOT suppress
— delete the import, or add to `__all__` for `__init__.py` re-exports).

IDE `★` dead-code hints flag *every* unused name regardless of prefix —
a separate always-on channel. When `<new-diagnostics>` shows only `★`
items, verify against the CLI before editing — frequently IDE-only noise.

