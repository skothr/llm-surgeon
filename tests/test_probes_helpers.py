"""Unit tests for pure helpers inside the generate WS route.

No model or WebSocket machinery required — exercises just the logic the
backend uses to decide what to emit when a stop sequence fires.
"""

import pytest

from gui.backend.routes.probes import _check_stop


class TestCheckStop:
    def test_no_stops_returns_chunk_verbatim(self):
        visible, hit, matched = _check_stop("hello", " world", [])
        assert visible == " world"
        assert hit is False
        assert matched == ""

    def test_no_match_returns_chunk_verbatim(self):
        visible, hit, matched = _check_stop("hello", " world", ["\n\n", "END"])
        assert visible == " world"
        assert hit is False
        assert matched == ""

    def test_stop_entirely_within_new_chunk(self):
        # Token contains " world\n\n" — everything before "\n\n" should leak
        # through, the stop itself must not.
        visible, hit, matched = _check_stop("hello", " world\n\nmore", ["\n\n"])
        assert visible == " world"
        assert hit is True
        assert matched == "\n\n"

    def test_stop_straddles_token_boundary(self):
        # Accumulated already ends with "\n", new chunk starts with "\n".
        # The stop string "\n\n" only completes now.
        visible, hit, matched = _check_stop("hello\n", "\nworld", ["\n\n"])
        assert visible == ""
        assert hit is True
        assert matched == "\n\n"

    def test_earliest_match_wins(self):
        # Both "END" and "STOP" appear; "END" is earlier.
        visible, hit, matched = _check_stop("", "foo END bar STOP", ["STOP", "END"])
        assert visible == "foo "
        assert hit is True
        assert matched == "END"

    def test_stop_matches_exactly_at_boundary(self):
        # Stop starts exactly where accumulated ends — visible should be empty.
        visible, hit, matched = _check_stop("hello", "STOPnow", ["STOP"])
        assert visible == ""
        assert hit is True
        assert matched == "STOP"

    def test_empty_new_chunk_no_hit(self):
        visible, hit, matched = _check_stop("hello", "", ["STOP"])
        assert visible == ""
        assert hit is False
        assert matched == ""


class TestDecodeFirstTokenLeadingSpace:
    """Regression: the WS generate handler decodes generated IDs
    incrementally and slices off what was already emitted.
    SentencePiece-based tokenizers (Llama family) strip the leading
    ▁→space marker whenever decode() runs on a buffer that begins at
    position 0 — neither skip_special_tokens=True nor
    clean_up_tokenization_spaces=False prevents this. The fix decodes
    `[last_prompt_token, *gen_tokens]` so the SP decoder has prior-
    token context, then slices off the boundary token's own decoded
    prefix to recover just the gen text. These tests pin the
    contract: the first gen token's leading space MUST survive.
    """

    def _tokenizer(self):
        from transformers import AutoTokenizer

        try:
            return AutoTokenizer.from_pretrained(
                "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                local_files_only=True,
            )
        except Exception as e:
            pytest.skip(f"TinyLlama tokenizer not cached: {e}")

    def test_naive_decode_loses_leading_space(self):
        """Sanity check: decoding gen-only IDs from position 0 strips
        the leading space — confirms the bug exists and the boundary-
        token workaround below is solving the right problem."""
        tok = self._tokenizer()
        ids = tok.encode(" Paris", add_special_tokens=False)
        decoded = tok.decode(ids, skip_special_tokens=True)
        assert not decoded.startswith(" "), (
            f"expected the buggy behavior on naive decode, got {decoded!r}"
        )

    def test_boundary_token_decode_preserves_leading_space(self):
        """Decoding `[boundary_id, *gen_ids]` and slicing off the
        boundary's own decoded prefix yields the gen text WITH its
        leading SP space, because the SP decoder no longer sees the
        gen IDs at position 0."""
        tok = self._tokenizer()
        prompt_ids = tok.encode("The capital of France is", add_special_tokens=False)
        gen_ids = tok.encode(" Paris", add_special_tokens=False)
        boundary_id = prompt_ids[-1]
        boundary_only = tok.decode([boundary_id], skip_special_tokens=True)
        boundary_plus = tok.decode([boundary_id, *gen_ids], skip_special_tokens=True)
        gen_text = boundary_plus[len(boundary_only):]
        assert gen_text.startswith(" "), (
            f"expected leading space preserved via boundary-token decode, got {gen_text!r}"
        )
        assert gen_text == " Paris"

    def test_incremental_slice_reconstructs_full_text(self):
        """Per-step incremental decode (the actual handler pattern)
        must reproduce the full decoded text when chunks are
        concatenated, with the first chunk carrying the leading
        space."""
        tok = self._tokenizer()
        prompt_ids = tok.encode("The capital of France is", add_special_tokens=False)
        gen_ids = tok.encode(" Paris is the capital.", add_special_tokens=False)
        boundary_id = prompt_ids[-1]
        boundary_only = tok.decode([boundary_id], skip_special_tokens=True)
        prev = ""
        chunks = []
        for k in range(1, len(gen_ids) + 1):
            full = tok.decode(
                [boundary_id, *gen_ids[:k]], skip_special_tokens=True
            )[len(boundary_only):]
            chunks.append(full[len(prev):])
            prev = full
        assert "".join(chunks) == prev
        assert prev.startswith(" "), (
            f"expected leading-space preserved across full decode, got {prev!r}"
        )
