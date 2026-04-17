# Phase 1 — Model Loading Refactor (Design Spec)

**Date:** 2026-04-17
**Status:** Approved for planning
**Roadmap ref:** `project_llm_surgeon_roadmap.md` — Phase 1

## Problem

`llm_surgeon/surgery.py::load_model` resolves cached HuggingFace models by
hand: `_snapshot_dir()` reads `refs/main` directly, `convert_to_safetensors`
reshapes the cache in place, and `HF_HUB_OFFLINE` is mutated via `os.environ`
around the load. This duplicates logic HuggingFace already provides, races the
cache GC (dangling refs), loses per-call `revision=` pinning, and leaks a
disk-level conversion step into the GUI as a user-facing button.

## Goal

Replace the hand-rolled cache resolution with the stock HF Hub pattern:

```python
AutoModelForCausalLM.from_pretrained(
    model_id,
    cache_dir=MODEL_CACHE_DIR,
    revision=revision,
    local_files_only=cached,
    use_safetensors=True,
    ...mode-specific kwargs...,
)
```

…and delete every piece of code that exists only to work around the old
approach.

## Non-goals

- GGUF/Ollama branch in `load_model` — audited clean; untouched.
- Phase 2 (lm-eval-harness) and Phase 3 (activation patching) work.
- `_quantize_in_place` / BitsAndBytes logic — only the load step changes.

## Decisions

Seven brainstorming decisions drove this design; recording them here so the
plan and implementation phases don't relitigate:

| # | Question | Choice | Why |
|---|---|---|---|
| 1 | Revision pinning policy | Default `revision=None`, optional kwarg | Matches typical HF usage; reproducibility is opt-in per call. |
| 2 | Fate of `convert_to_safetensors` | Delete endpoint + button | Modern HF caches ship safetensors; `use_safetensors=True` auto-handles fallback. |
| 3 | Replacement for "is cached?" check | `huggingface_hub.try_to_load_from_cache("config.json")` | Stock HF primitive; handles stale-ref edge cases `_snapshot_dir` misses. |
| 4 | Offline-mode plumbing | Per-call `local_files_only=True` when cached | Scoped to the call; no global env-var mutation. |
| 5 | GGUF/Ollama branch | Out of scope, audit only | `gguf_reader.py` has zero matches for `_snapshot_dir`/`MODEL_CACHE_DIR`/`cache_dir`. |
| 6 | Tokenizer load | Parallel `AutoTokenizer.from_pretrained` call | Tiny duplication, no helper needed yet. |
| 7 | Test scope | Unit (kwarg routing, mocked) + integration (TinyLlama) | Unit catches regressions fast; integration catches real HF-cache behavior. |

## New public API

```python
def load_model(
    model_id: str,
    mode: str = "nf4",
    *,
    revision: str | None = None,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
```

All existing call sites pass `(model_id, mode=...)` — unchanged. `revision` is
an opt-in keyword-only kwarg.

## Replacement load flow

Inside `load_model`, after the (untouched) Ollama branch:

```python
is_local = os.path.isdir(model_id)
cached = (not is_local) and _is_cached(model_id)

common_kwargs = {
    "use_safetensors": True,
    "revision": revision,
    **({} if is_local else {
        "cache_dir": MODEL_CACHE_DIR,
        "local_files_only": cached,
    }),
}

# Mode dispatch unchanged — each branch merges common_kwargs with
# its mode-specific kwargs (quantization_config, torch_dtype, device_map).
```

Tokenizer loaded with the same `common_kwargs` minus `use_safetensors`.

### Key behaviors

- `use_safetensors=True` prefers safetensors shards, falls back to `.bin`.
  Makes the old custom `.bin → safetensors` step unnecessary.
- `local_files_only=True` only when `_is_cached()` confirmed presence —
  avoids the "caller sees offline after surgery load" latching bug that the
  current `finally` block papers over.
- No `HF_HUB_OFFLINE` env-var mutation.

## New helper `_is_cached`

```python
def _is_cached(model_id: str, cache_dir: str | None = None) -> bool:
    from huggingface_hub import try_to_load_from_cache
    path = try_to_load_from_cache(
        model_id, filename="config.json",
        cache_dir=cache_dir or MODEL_CACHE_DIR,
    )
    return path is not None
```

Exported from `surgery.py` so the GUI can import it. Uses `config.json` as a
universal, tiny sentinel file every HF model ships.

GUI swaps both `_snapshot_dir()` probes (sessions.py:675 revert, :851 clone)
for `_is_cached()` — semantic unchanged, delegation corrected.

## Deletions

### `llm_surgeon/surgery.py` (~95 lines)

- `_snapshot_dir` (19 lines)
- `_has_safetensors` (8 lines)
- `convert_to_safetensors` (47 lines)
- `HF_HUB_OFFLINE` save/restore wrapping inside `load_model` (10 lines)
- Snapshot-dir load-path branching inside `load_model` (~10 lines)

### `gui/backend/routes/sessions.py`

- `_has_safetensors_cached` helper (5 lines)
- `ConvertRequest` model + `POST /models/convert-safetensors` endpoint (~25 lines)
- Call-site imports of `_has_safetensors`, `convert_to_safetensors`, `_snapshot_dir`
- `_snapshot_dir(info.model_id)` → `_is_cached(info.model_id)` at lines 675, 851
- `"safetensors": _has_safetensors_cached(model_id)` field at line 60 — drop
  it from the payload (nothing meaningful to report post-refactor)

### `gui/frontend`

- "Convert to safetensors" button, handler, and any adjacent disabled-state /
  toast plumbing. Exact files enumerated during planning via grep.

## Tests

### Unit (new, mocked)

- `test_load_model_hub_id_not_cached` — assert `cache_dir`, `revision=None`,
  `local_files_only=False`, `use_safetensors=True` passed to
  `from_pretrained`.
- `test_load_model_hub_id_cached` — patch `_is_cached` → True; assert
  `local_files_only=True`.
- `test_load_model_local_path` — `os.path.isdir` True; no `cache_dir` or
  `local_files_only` kwargs.
- `test_load_model_with_revision` — `revision="abc123"` propagates.
- `test_is_cached_returns_false_for_uncached` — fresh cache dir, unknown
  repo, `_is_cached` returns False.
- `test_is_cached_returns_true_after_seed` — seed cache with a minimal
  `config.json` under the HF layout, assert True.

### Integration (new)

- `test_load_model_tinyllama_fp16` — real load against the TinyLlama cache
  (CLAUDE.md default dev model); assert tokenizer + model return; assert
  `model.config.num_hidden_layers == 22`.
- `test_load_model_tinyllama_nf4` — same, `mode="nf4"`; assert quantized
  Linear layers present.

Guarded with `@pytest.mark.skipif(not tinyllama_cached)` so CI without the
cache degrades gracefully.

### Updated

- `test_surgery.py:366` — existing `MODEL_CACHE_DIR` assertion keeps working.
- Any test referencing `convert_to_safetensors` / `_snapshot_dir` /
  `_has_safetensors` → delete.

## Migration & risk

### Migration (zero user action)

- Caches with safetensors: load unchanged (preferred path).
- Caches that previously went through `convert_to_safetensors`: safetensors
  still present; load unchanged.
- `.bin`-only caches: HF falls back automatically. Slightly slower; correct.
  User may delete and re-fetch for the speedup.

### Risks

- `try_to_load_from_cache` returns a string, `None`, or the sentinel
  `_CACHED_NO_EXIST` (negative-cache marker). `is not None` accepts the
  sentinel — fine for our use, since the sentinel implies someone *tried* to
  cache. Unit test pins this behavior.
- GUI frontend may have adjacent UI state for the convert button (spinner,
  toasts). Plan phase enumerates every removal via grep.
- No known risk to `revert_surgery` / `clone_session` — `_is_cached`
  semantics match `_snapshot_dir is not None`.

### Rollback

Single-commit revert restores prior state. Cache contents on disk untouched.

## Commit strategy

Three commits, each self-contained and tests-green:

1. `refactor(surgery): replace _snapshot_dir with stock HF cache API` —
   all `surgery.py` + `test_surgery.py` changes.
2. `feat(gui/backend): adopt _is_cached; drop safetensors-convert endpoint` —
   backend `sessions.py` edits.
3. `feat(gui/frontend): remove convert-to-safetensors button` — frontend
   cleanup.

Order is load-bearing: backend can't import `_is_cached` until #1 lands;
frontend won't compile cleanly against the old response shape if #2 changes
the cache-info payload before #3.

## Expected outcome

- ~100 LOC removed from `surgery.py`, one GUI endpoint removed, one
  frontend button removed.
- No runtime behavior change for users with healthy caches.
- Reproducibility upgrade available via opt-in `revision=` kwarg.
- Cleaner foundation for Phase 2 (`benchmark_harness.py`) which will call
  `load_model` with pinned revisions.
