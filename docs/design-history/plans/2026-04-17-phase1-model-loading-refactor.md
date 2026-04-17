# Phase 1 Model-Loading Refactor — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `_snapshot_dir`-based cache resolution and the custom `.bin → safetensors` conversion in `llm_surgeon/surgery.py` with the stock HuggingFace Hub cache API (`try_to_load_from_cache`, `cache_dir=`, `revision=`, `local_files_only=`, `use_safetensors=True`). Delete the GUI's now-unused `/models/convert-safetensors` endpoint and its `safetensors` badge.

**Architecture:** Four code locations change: `llm_surgeon/surgery.py` (add `_is_cached`, simplify `load_model`, delete legacy helpers); `gui/backend/routes/sessions.py` (swap `_snapshot_dir` for `_is_cached`, drop the convert endpoint and the `safetensors` payload field); `gui/frontend/src/types/api.ts` + `components/SessionsPanel.tsx` (drop the `safetensors` type field and its badge); `tests/test_surgery.py` (add unit + integration coverage for the new load flow). Work ships as three self-contained commits so each one stays green; a fourth commit bumps memory with Phase-1 completion state.

**Tech Stack:** Python 3 · PyTorch · transformers · huggingface_hub · pytest · FastAPI · React/TypeScript · Vite · Playwright

**Spec:** `testing/docs/superpowers/specs/2026-04-17-phase1-model-loading-design.md`

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `testing/llm_surgeon/surgery.py` | Modify | Add `_is_cached`; rewrite `load_model` load path; delete `_snapshot_dir`, `_has_safetensors`, `convert_to_safetensors`, `HF_HUB_OFFLINE` env mutation |
| `testing/tests/test_surgery.py` | Modify | Add `TestIsCached` + `TestLoadModel` (unit, mocked); add TinyLlama integration tests; delete/update tests referencing removed helpers |
| `testing/gui/backend/routes/sessions.py` | Modify | Replace `_snapshot_dir` guards with `_is_cached`; delete `_has_safetensors_cached`, `ConvertRequest`, `/models/convert-safetensors`; drop `safetensors` field from `_hf_model_meta` |
| `testing/gui/frontend/src/types/api.ts` | Modify | Remove `safetensors: boolean` from `AvailableModel` |
| `testing/gui/frontend/src/components/SessionsPanel.tsx` | Modify | Remove the `safetensors` span at :285 |
| `/home/skothr/.claude/projects/-home-ai-ai-projects-llm/memory/project_llm_surgeon_roadmap.md` | Modify | Update status line to record Phase 1 shipped |

---

## Task Ordering

Three commits in this order (spec §8 is load-bearing):

1. **Task 1** — `refactor(surgery): replace _snapshot_dir with stock HF cache API` (all `surgery.py` + `test_surgery.py` changes)
2. **Task 2** — `feat(gui/backend): adopt _is_cached; drop safetensors-convert endpoint` (backend `sessions.py` edits)
3. **Task 3** — `feat(gui/frontend): drop safetensors badge from SessionsPanel` (frontend cleanup — type + component)
4. **Task 4** — Update the roadmap memory to mark Phase 1 shipped (no commit — memory edit only)

Order matters because Task 2 imports `_is_cached` (introduced in Task 1) and Task 3 compiles against the response shape shipped in Task 2 (the `safetensors` field is dropped from the backend payload there).

---

## Task 1 — Surgery.py refactor (TDD)

**Files:**
- Modify: `testing/llm_surgeon/surgery.py`
- Modify: `testing/tests/test_surgery.py`

---

- [ ] **Step 1.1 — Write failing test for `_is_cached` on an empty cache**

Add to `testing/tests/test_surgery.py`:

```python
class TestIsCached:
    def test_returns_false_for_uncached(self, tmp_path):
        from llm_surgeon.surgery import _is_cached
        assert _is_cached("nonexistent/repo", cache_dir=str(tmp_path)) is False

    def test_returns_true_when_config_present(self, tmp_path):
        """Seed the HF cache layout with a minimal config.json and assert hit."""
        from llm_surgeon.surgery import _is_cached
        import json

        repo = "TinyOrg/TinyModel"
        slug = "models--TinyOrg--TinyModel"
        sha = "a" * 40
        model_dir = tmp_path / slug
        (model_dir / "snapshots" / sha).mkdir(parents=True)
        (model_dir / "refs").mkdir(parents=True)
        (model_dir / "refs" / "main").write_text(sha)
        cfg = model_dir / "snapshots" / sha / "config.json"
        cfg.write_text(json.dumps({"model_type": "llama"}))
        blob = model_dir / "blobs" / "dummy-blob"
        blob.parent.mkdir(parents=True, exist_ok=True)
        blob.write_text(json.dumps({"model_type": "llama"}))
        cfg.unlink()
        cfg.symlink_to(blob)

        assert _is_cached(repo, cache_dir=str(tmp_path)) is True
```

- [ ] **Step 1.2 — Run test to confirm it fails**

Run: `testing/.venv/bin/python -m pytest testing/tests/test_surgery.py::TestIsCached -v`

Expected: FAIL with `ImportError` — `_is_cached` does not exist yet.

- [ ] **Step 1.3 — Implement `_is_cached` in `surgery.py`**

Add to `testing/llm_surgeon/surgery.py` directly above the `_snapshot_dir` function:

```python
def _is_cached(model_id: str, cache_dir: str | None = None) -> bool:
    """True if a local HF cache has at least a config.json snapshot for model_id.

    Wraps huggingface_hub.try_to_load_from_cache. Used as the boolean probe
    that replaces _snapshot_dir for "is this model present locally?".
    """
    from huggingface_hub import try_to_load_from_cache
    path = try_to_load_from_cache(
        model_id, filename="config.json",
        cache_dir=cache_dir or MODEL_CACHE_DIR,
    )
    return path is not None
```

- [ ] **Step 1.4 — Run test to confirm it passes**

Run: `testing/.venv/bin/python -m pytest testing/tests/test_surgery.py::TestIsCached -v`

Expected: both tests PASS.

- [ ] **Step 1.5 — Write failing unit tests for `load_model` kwarg routing**

Add to `testing/tests/test_surgery.py`:

```python
class TestLoadModelKwargs:
    """Verify load_model passes the right kwargs to from_pretrained.

    Mocks AutoModelForCausalLM and AutoTokenizer to capture the kwargs
    without actually loading weights.
    """

    def _install_mocks(self, monkeypatch):
        captured = {}

        class _FakeModel:
            pass

        def fake_model_from_pretrained(load_id, **kwargs):
            captured["model_load_id"] = load_id
            captured["model_kwargs"] = kwargs
            return _FakeModel()

        def fake_tok_from_pretrained(load_id, **kwargs):
            captured["tok_load_id"] = load_id
            captured["tok_kwargs"] = kwargs
            return _FakeModel()

        monkeypatch.setattr(
            "llm_surgeon.surgery.AutoModelForCausalLM.from_pretrained",
            fake_model_from_pretrained,
        )
        monkeypatch.setattr(
            "llm_surgeon.surgery.AutoTokenizer.from_pretrained",
            fake_tok_from_pretrained,
        )
        return captured

    def test_hub_id_not_cached(self, monkeypatch):
        from llm_surgeon import surgery
        captured = self._install_mocks(monkeypatch)
        monkeypatch.setattr(surgery, "_is_cached", lambda *a, **kw: False)

        surgery.load_model("Org/Model", mode="fp16")

        assert captured["model_load_id"] == "Org/Model"
        mkw = captured["model_kwargs"]
        assert mkw["cache_dir"] == surgery.MODEL_CACHE_DIR
        assert mkw["revision"] is None
        assert mkw["local_files_only"] is False
        assert mkw["use_safetensors"] is True
        assert mkw["torch_dtype"] is __import__("torch").float16

    def test_hub_id_cached(self, monkeypatch):
        from llm_surgeon import surgery
        captured = self._install_mocks(monkeypatch)
        monkeypatch.setattr(surgery, "_is_cached", lambda *a, **kw: True)

        surgery.load_model("Org/Model", mode="fp16")

        assert captured["model_kwargs"]["local_files_only"] is True

    def test_local_path_no_cache_kwargs(self, monkeypatch, tmp_path):
        from llm_surgeon import surgery
        # Create a dummy local path so os.path.isdir returns True
        (tmp_path / "weights").mkdir()
        captured = self._install_mocks(monkeypatch)

        surgery.load_model(str(tmp_path / "weights"), mode="fp16")

        mkw = captured["model_kwargs"]
        assert "cache_dir" not in mkw
        assert "local_files_only" not in mkw
        # revision + use_safetensors still propagate for local paths
        assert mkw["use_safetensors"] is True

    def test_revision_propagates(self, monkeypatch):
        from llm_surgeon import surgery
        captured = self._install_mocks(monkeypatch)
        monkeypatch.setattr(surgery, "_is_cached", lambda *a, **kw: False)

        surgery.load_model("Org/Model", mode="fp16", revision="abc123")

        assert captured["model_kwargs"]["revision"] == "abc123"
        assert captured["tok_kwargs"]["revision"] == "abc123"
```

- [ ] **Step 1.6 — Run the new tests to confirm they fail**

Run: `testing/.venv/bin/python -m pytest testing/tests/test_surgery.py::TestLoadModelKwargs -v`

Expected: all four tests FAIL (signature of `load_model` doesn't yet accept `revision`; `_is_cached` not yet wired into `load_model`; env-var mutation still present).

- [ ] **Step 1.7 — Rewrite `load_model` in `surgery.py`**

Replace the current `load_model` function (lines 748–855) with this version. The Ollama branch (`_is_ollama_id` block) is preserved verbatim; only the post-Ollama HF/local branch changes.

```python
def load_model(
    model_id: str,
    mode: str = "nf4",
    *,
    revision: str | None = None,
) -> Tuple:
    """Load a model and tokenizer.

    Modes:
        nf4:      4-bit NormalFloat on GPU (smallest, for surgery/inspection)
        int8:     8-bit LLM.int8() on GPU (balanced quality/memory)
        fp16:     half-precision with auto device map
        fp32:     full precision with auto device map
        fp32-cpu: full precision forced to CPU (for export)

    Supports HuggingFace Hub IDs, local paths, and Ollama model IDs
    (e.g. 'tinyllama:latest'). Ollama models are loaded from GGUF and
    dequantized into standard HuggingFace models.

    Args:
        revision: Optional HF Hub commit SHA / branch / tag. Pass to pin an
            experiment to an exact model snapshot. Ignored for local paths
            and Ollama IDs.
    """
    mode = _MODE_ALIASES.get(mode, mode)
    if mode not in VALID_MODES:
        raise ValueError(f"Unknown mode: '{mode}'. Must be one of {sorted(VALID_MODES)}.")

    # Try Ollama resolution for non-HF, non-local model IDs
    if _is_ollama_id(model_id):
        from .gguf_reader import resolve_ollama_blob, load_gguf_as_hf
        blob = resolve_ollama_blob(model_id)
        if blob is not None:
            _GGUF_DTYPE = {
                "nf4": torch.bfloat16, "int8": torch.bfloat16,
                "bf16": torch.bfloat16, "fp16": torch.float16,
                "fp32": torch.float32, "fp32-cpu": torch.float32,
            }
            model, tokenizer = load_gguf_as_hf(blob, dtype=_GGUF_DTYPE.get(mode, torch.float16))
            if mode == "nf4":
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                )
                model = _quantize_in_place(model, bnb_config)
            elif mode == "int8":
                bnb_config = BitsAndBytesConfig(load_in_8bit=True)
                model = _quantize_in_place(model, bnb_config)
            return model, tokenizer

    is_local = os.path.isdir(model_id)
    cached = (not is_local) and _is_cached(model_id)

    common_kwargs: Dict[str, Any] = {
        "use_safetensors": True,
        "revision": revision,
    }
    if not is_local:
        common_kwargs["cache_dir"] = MODEL_CACHE_DIR
        common_kwargs["local_files_only"] = cached

    if mode == "nf4":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=bnb_config, device_map="auto",
            **common_kwargs,
        )
    elif mode == "int8":
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=bnb_config, device_map="auto",
            **common_kwargs,
        )
    elif mode == "bf16":
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, **common_kwargs,
        )
    elif mode == "fp16":
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16, **common_kwargs,
        )
    elif mode == "fp32":
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float32, **common_kwargs,
        )
    elif mode == "fp32-cpu":
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float32, device_map="cpu",
            **common_kwargs,
        )
    else:
        raise ValueError(f"Unknown mode: '{mode}'")

    # AutoTokenizer does not accept use_safetensors — strip it.
    tok_kwargs = {k: v for k, v in common_kwargs.items() if k != "use_safetensors"}
    tokenizer = AutoTokenizer.from_pretrained(model_id, **tok_kwargs)

    return model, tokenizer
```

- [ ] **Step 1.8 — Delete the legacy helpers from `surgery.py`**

Delete these three top-level functions (they no longer have callers after Step 1.7):

- `_snapshot_dir` (currently lines 22–40)
- `_has_safetensors` (currently lines 43–50)
- `convert_to_safetensors` (currently lines 53–99)

Leave `MODEL_CACHE_DIR` in place — it's still used as the default for `cache_dir=` and imported by `test_surgery.py:367`.

- [ ] **Step 1.9 — Run `TestLoadModelKwargs` to confirm it passes**

Run: `testing/.venv/bin/python -m pytest testing/tests/test_surgery.py::TestLoadModelKwargs -v`

Expected: all four tests PASS.

- [ ] **Step 1.10 — Write failing TinyLlama integration tests**

Append to `testing/tests/test_surgery.py`. These rely on the TinyLlama cache documented in `CLAUDE.md § Primary dev models`; a `skipif` guards CI/offline environments without the cache.

```python
def _tinyllama_cached() -> bool:
    from llm_surgeon.surgery import _is_cached
    return _is_cached("TinyLlama/TinyLlama-1.1B-Chat-v1.0")


@pytest.mark.skipif(not _tinyllama_cached(), reason="TinyLlama not cached")
class TestLoadModelIntegration:
    def test_tinyllama_fp16(self):
        from llm_surgeon.surgery import load_model
        model, tokenizer = load_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0", mode="fp16")
        assert tokenizer is not None
        assert model.config.num_hidden_layers == 22
        assert model.config.hidden_size == 2048

    def test_tinyllama_nf4(self):
        import bitsandbytes as bnb  # quantization check needs the library
        from llm_surgeon.surgery import load_model
        model, tokenizer = load_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0", mode="nf4")
        assert tokenizer is not None
        # At least one Linear in an attention block should be Linear4bit.
        attn = model.model.layers[0].self_attn
        assert isinstance(attn.q_proj, bnb.nn.Linear4bit)
```

- [ ] **Step 1.11 — Run the integration tests**

Run: `testing/.venv/bin/python -m pytest testing/tests/test_surgery.py::TestLoadModelIntegration -v`

Expected: both tests PASS (assuming TinyLlama is cached; otherwise skipped).

Note: `nf4` needs CUDA. If the runner has no GPU, the test may fail at `device_map="auto"` placement — in that case, also skip with `@pytest.mark.skipif(not torch.cuda.is_available(), ...)` and re-run.

- [ ] **Step 1.12 — Delete the obsolete `test_export` test**

`testing/tests/test_export.py` may reference `convert_to_safetensors`. Search and update:

Run: `grep -nE 'convert_to_safetensors|_snapshot_dir|_has_safetensors' testing/tests/`

For every match, either delete the referring test or rewrite it against the new API (most likely delete — these helpers no longer exist). Remove related imports at the top of the file.

- [ ] **Step 1.13 — Full test suite sanity check**

Run: `testing/.venv/bin/python -m pytest testing/tests/ -v`

Expected: all tests PASS (or skip cleanly if GPU/cache absent). Fix any regression surfaced.

- [ ] **Step 1.14 — Type-check surgery.py and tests**

Run: `testing/.venv/bin/python -m pyright testing/llm_surgeon/surgery.py testing/tests/test_surgery.py`

Expected: zero errors, zero warnings, zero informations. Fix any surfaced diagnostics before committing (per CLAUDE.md § Type Checking).

- [ ] **Step 1.15 — Commit**

```bash
cd /home/ai/ai-projects/llm
git add testing/llm_surgeon/surgery.py testing/tests/test_surgery.py testing/tests/test_export.py
git commit -m "$(cat <<'EOF'
refactor(surgery): replace _snapshot_dir with stock HF cache API

Swap hand-rolled snapshot-dir resolution and in-place .bin→safetensors
conversion for stock from_pretrained(cache_dir=, revision=,
local_files_only=, use_safetensors=True). Add _is_cached() helper
backed by huggingface_hub.try_to_load_from_cache. Drop HF_HUB_OFFLINE
env-var juggling — scope is now per-call.

Net -~95 LOC in surgery.py; new load_model signature adds opt-in
revision= kwarg for reproducibility.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

Run: `git status` — expected clean tree afterwards.

---

## Task 2 — Backend sessions.py cleanup

**Files:**
- Modify: `testing/gui/backend/routes/sessions.py`

---

- [ ] **Step 2.1 — Delete `_has_safetensors_cached` helper**

In `testing/gui/backend/routes/sessions.py`, delete the helper at lines 497–501:

```python
def _has_safetensors_cached(model_id: str) -> bool:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
    from llm_surgeon.surgery import _has_safetensors
    return _has_safetensors(model_id, str(MODELS_CACHE))
```

- [ ] **Step 2.2 — Drop `safetensors` field from `_hf_model_meta`**

At lines 58–60, change:

```python
def _hf_model_meta(model_id: str) -> dict:
    meta: dict = {"model_id": model_id, "source": "huggingface",
                  "safetensors": _has_safetensors_cached(model_id)}
```

to:

```python
def _hf_model_meta(model_id: str) -> dict:
    meta: dict = {"model_id": model_id, "source": "huggingface"}
```

- [ ] **Step 2.3 — Delete `ConvertRequest` and `convert_model_safetensors` endpoint**

Delete the entire block currently spanning `class ConvertRequest` through the end of `convert_model_safetensors` (surrounding `@router.post("/models/convert-safetensors")`). This sits immediately after `_has_safetensors_cached` in the current file.

- [ ] **Step 2.4 — Replace `_snapshot_dir` call at the revert endpoint**

Find the import+call around line 663:

```python
from llm_surgeon.surgery import _snapshot_dir
import shutil
import torch as _torch

async with info.lock:
    ...
    if _snapshot_dir(info.model_id):
```

Rewrite as:

```python
from llm_surgeon.surgery import _is_cached
import shutil
import torch as _torch

async with info.lock:
    ...
    if _is_cached(info.model_id):
```

- [ ] **Step 2.5 — Replace `_snapshot_dir` call at the clone endpoint**

Find the call around line 850:

```python
from llm_surgeon.surgery import _snapshot_dir
if _snapshot_dir(info.model_id):
```

Rewrite as:

```python
from llm_surgeon.surgery import _is_cached
if _is_cached(info.model_id):
```

- [ ] **Step 2.6 — Inspect `_scan_executor` — leave it alone unless orphaned**

Scan the file for remaining `_scan_executor.submit` or `run_in_executor(_scan_executor, ...)` uses:

Run: `grep -n '_scan_executor' testing/gui/backend/routes/sessions.py`

If uses remain (model-cache scan, GGUF header parsing), leave the executor as-is. If nothing uses it after the delete in Step 2.3, delete its definition at the top of the file.

- [ ] **Step 2.7 — Type-check the backend file**

Run: `testing/.venv/bin/python -m pyright testing/gui/backend/routes/sessions.py`

Expected: zero diagnostics. Fix any surfaced by the deletes (usually an orphaned import — `field_validator`, `BaseModel`, `Path` etc. may still be used elsewhere; only remove if pyright flags them).

- [ ] **Step 2.8 — Boot the backend for a smoke check**

The session-management surface is not covered by the Playwright smoke suite (spec §7 noted it). Run a quick manual boot to verify no import-time crash:

```bash
cd /home/ai/ai-projects/llm/testing
./.venv/bin/python -c "from gui.backend.routes import sessions; print('OK:', sessions.router.routes[0].path)"
```

Expected: prints `OK: ...` and exits 0. Any ImportError means a deletion nicked a live caller — fix before committing.

- [ ] **Step 2.9 — Commit**

```bash
cd /home/ai/ai-projects/llm
git add testing/gui/backend/routes/sessions.py
git commit -m "$(cat <<'EOF'
feat(gui/backend): adopt _is_cached; drop safetensors-convert endpoint

Route cache-presence probes through the new _is_cached() helper instead
of the deleted _snapshot_dir. Remove the /models/convert-safetensors
endpoint and its pydantic model — modern HF caches ship safetensors and
from_pretrained(use_safetensors=True) handles fallback. Drop the
now-meaningless safetensors boolean from _hf_model_meta.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

Run: `git status` — expected clean.

---

## Task 3 — Frontend safetensors-badge removal

**Files:**
- Modify: `testing/gui/frontend/src/types/api.ts`
- Modify: `testing/gui/frontend/src/components/SessionsPanel.tsx`

No separate convert-button exists in the frontend — verified by grep during planning. The only UI surface that referenced safetensors is the info-badge span at `SessionsPanel.tsx:285`.

---

- [ ] **Step 3.1 — Remove `safetensors` field from `AvailableModel`**

In `testing/gui/frontend/src/types/api.ts`, delete line 138:

```ts
  safetensors: boolean;
```

The remaining `AvailableModel` fields stay untouched.

- [ ] **Step 3.2 — Remove the safetensors badge from `SessionsPanel.tsx`**

In `testing/gui/frontend/src/components/SessionsPanel.tsx`, delete line 285:

```tsx
{selectedModel.safetensors && <span> | safetensors</span>}
```

Verify nothing else in the file references `selectedModel.safetensors`:

Run: `grep -n 'safetensors' testing/gui/frontend/src/components/SessionsPanel.tsx`

Expected: no matches.

- [ ] **Step 3.3 — Sanity grep the rest of the frontend**

Run: `grep -rn 'safetensors' testing/gui/frontend/src/`

Expected: zero matches. If any remain, they were missed references — delete them.

- [ ] **Step 3.4 — Tier 1 type check**

Run:

```bash
cd /home/ai/ai-projects/llm/testing/gui/frontend
./node_modules/.bin/tsc --noEmit
```

Expected: exit 0, no diagnostics.

- [ ] **Step 3.5 — Tier 2 production build**

Run:

```bash
cd /home/ai/ai-projects/llm/testing/gui/frontend
./node_modules/.bin/vite build
```

Expected: exit 0, bundle written to `dist/`.

- [ ] **Step 3.6 — Tier 3 Playwright smoke suite**

Per CLAUDE.md: "Always run Tier 3 after UI or store changes."

Run:

```bash
cd /home/ai/ai-projects/llm/testing/gui/frontend
npm run e2e
```

Expected: 9 tests PASS. If vite/playwright tooling hits the sandbox at `/dev/urandom`, re-run with `dangerouslyDisableSandbox: true` (per CLAUDE.md note on node crypto).

- [ ] **Step 3.7 — Commit**

```bash
cd /home/ai/ai-projects/llm
git add testing/gui/frontend/src/types/api.ts testing/gui/frontend/src/components/SessionsPanel.tsx
git commit -m "$(cat <<'EOF'
feat(gui/frontend): drop safetensors badge from SessionsPanel

Backend no longer reports the safetensors boolean — from_pretrained
handles the format transparently now. Remove the dead field from the
AvailableModel type and the info-badge span that consumed it.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

Run: `git status` — expected clean.

---

## Task 4 — Mark Phase 1 shipped in roadmap memory

**Files:**
- Modify: `/home/skothr/.claude/projects/-home-ai-ai-projects-llm/memory/project_llm_surgeon_roadmap.md`

No git commit — memory lives outside the repo.

---

- [ ] **Step 4.1 — Update the `### Status` section**

Edit the roadmap memory's `### Status` section (currently: "2026-04-17: plan saved. No phases started yet.") to read:

```
- 2026-04-17: plan saved.
- 2026-04-17: Phase 1 shipped — _snapshot_dir/convert_to_safetensors removed, replaced by stock HF cache API + _is_cached helper. Spec: testing/docs/superpowers/specs/2026-04-17-phase1-model-loading-design.md. Commits: <sha1>, <sha2>, <sha3>. Next up: Phase 2 (lm-eval-harness integration).
```

Substitute the actual commit SHAs from Tasks 1–3 (`git log --oneline -3`).

- [ ] **Step 4.2 — Verify the MEMORY.md index is already accurate**

Check that `MEMORY.md` does not need updating:

Run: `grep -n llm_surgeon_roadmap /home/skothr/.claude/projects/-home-ai-ai-projects-llm/memory/MEMORY.md`

Expected: one entry pointing to `project_llm_surgeon_roadmap.md`. No change needed; the link description is still accurate.

---

## Spec-Coverage Self-Review

Cross-check against `2026-04-17-phase1-model-loading-design.md`:

| Spec section | Task(s) |
|---|---|
| §2 New public API (`load_model` signature + `revision=` kwarg) | 1.7 |
| §3 Replacement load flow (common_kwargs, `use_safetensors=True`, `local_files_only=cached`, no env-var) | 1.7 |
| §4 `_is_cached` helper | 1.3 |
| §5 Deletions in `surgery.py` | 1.8 |
| §5 Deletions in `gui/backend` (`_has_safetensors_cached`, `ConvertRequest`, convert endpoint, safetensors payload field) | 2.1 / 2.2 / 2.3 |
| §5 Deletions in `gui/frontend` | 3.1 / 3.2 |
| §6 Unit tests (kwarg routing, revision, local-path) | 1.5 |
| §6 `_is_cached` unit tests | 1.1 |
| §6 TinyLlama integration tests | 1.10 |
| §6 Cleanup of `convert_to_safetensors` / `_snapshot_dir` references in tests | 1.12 |
| §7 Migration/risk (no user-action path, rollback) | Addressed by per-commit tests (1.13, 3.4-3.6) and commit-granularity structure |
| §8 Three-commit strategy | Task 1 / Task 2 / Task 3 commits |
| Roadmap memory status bump | Task 4 |

No placeholders (`TBD`, `TODO`, "similar to…", "add validation"). All types (`_is_cached: bool`, `revision: str | None`, `common_kwargs: Dict[str, Any]`) match between definition (1.3, 1.7) and usage (1.5, 2.4, 2.5). Commit SHAs in Task 4 are correctly left as substitution markers for resolution at execute time.

---

## Execution Handoff

Plan complete and saved to `testing/docs/superpowers/plans/2026-04-17-phase1-model-loading-refactor.md`. Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
