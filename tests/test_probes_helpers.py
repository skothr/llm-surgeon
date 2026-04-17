"""Unit tests for pure helpers inside the generate WS route.

No model or WebSocket machinery required — exercises just the logic the
backend uses to decide what to emit when a stop sequence fires.
"""

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
