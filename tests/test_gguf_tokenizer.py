"""GGUF tokenizer reconstruction: gguf_reader._build_tokenizer and the
tokenizer half of gguf_writer.export_hf_to_gguf.

References are independent of the code under test: a SentencePiece model
for the "llama" vocab type, and a Llama-3-style byte-level BPE built with
the ``tokenizers`` library for the "gpt2" vocab type.
"""

import json
import os

import pytest
from transformers import LlamaConfig, LlamaForCausalLM

from llm_surgeon.gguf_reader import GGUFFile, _build_tokenizer, load_gguf_as_hf

# Split pattern of Meta-Llama-3's tokenizer.json, copied here rather than
# imported so the test does not share the constant it checks.
LLAMA3_PATTERN = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)

CORPUS_LINES = [
    "The quick brown fox jumps over the lazy dog.",
    "Tokenization of unbelievable words, numbers like 3.14159 and 2718,",
    "and repeated letters xxxx yyyy. Hello world! hello there, general.",
    "It's a test: we'll see what they've done; I'm sure you'd agree.",
] * 50

TEXTS = [
    "Hello world",
    " leading space",
    "Hello  world\n\nnew",
    "unbelievable tokenization",
    "a\tb",
    "3.14159 is pi",
    "x" * 40,
    "It's what we'll do",
    "Ωmega \U0001f999",
    "  two leading",
    "trailing ",
]


# ── SentencePiece ("llama") vocab ─────────────────────────────────────

@pytest.fixture(scope="module")
def spm_processor(tmp_path_factory):
    spm = pytest.importorskip("sentencepiece")
    d = tmp_path_factory.mktemp("spm")
    corpus = d / "corpus.txt"
    corpus.write_text("\n".join(CORPUS_LINES) + "\n")
    spm.SentencePieceTrainer.train(  # pyright: ignore[reportAttributeAccessIssue]
        input=str(corpus),
        model_prefix=str(d / "m"),
        vocab_size=400,
        model_type="bpe",
        byte_fallback=True,
        character_coverage=1.0,
        num_threads=1,
        add_dummy_prefix=True,
        remove_extra_whitespaces=False,
        normalization_rule_name="identity",
    )
    return spm.SentencePieceProcessor(model_file=str(d / "m.model"))  # pyright: ignore[reportCallIssue]


def _spm_meta(sp, **extra) -> dict:
    """GGUF tokenizer metadata as convert_hf_to_gguf.py writes it for a
    SentencePiece model: tokens, scores and types, no merges."""
    n = sp.get_piece_size()
    types = []
    for i in range(n):
        if sp.is_unknown(i):
            types.append(2)
        elif sp.is_control(i):
            types.append(3)
        elif sp.is_byte(i):
            types.append(6)
        else:
            types.append(1)
    meta = {
        "tokenizer.ggml.model": "llama",
        "tokenizer.ggml.tokens": [sp.id_to_piece(i) for i in range(n)],
        "tokenizer.ggml.scores": [sp.get_score(i) for i in range(n)],
        "tokenizer.ggml.token_type": types,
        "tokenizer.ggml.bos_token_id": sp.bos_id(),
        "tokenizer.ggml.eos_token_id": sp.eos_id(),
        "tokenizer.ggml.unknown_token_id": sp.unk_id(),
    }
    meta.update(extra)
    return meta


class TestSentencePieceVocab:
    @pytest.mark.parametrize("text", TEXTS)
    def test_ids_match_sentencepiece(self, spm_processor, text):
        tok = _build_tokenizer(_spm_meta(spm_processor))
        assert tok is not None
        assert tok.encode(text) == [spm_processor.bos_id()] + spm_processor.encode(text)

    @pytest.mark.parametrize("text", TEXTS)
    def test_decode_roundtrip(self, spm_processor, text):
        tok = _build_tokenizer(_spm_meta(spm_processor))
        assert tok is not None
        assert tok.decode(tok.encode(text), skip_special_tokens=True) == text

    def test_out_of_vocab_chars_use_byte_fallback(self, spm_processor):
        tok = _build_tokenizer(_spm_meta(spm_processor))
        assert tok is not None
        ids = tok.encode("日本 \U0001f999")
        assert spm_processor.unk_id() not in ids
        assert "<0xF0>" in tok.convert_ids_to_tokens(ids)

    def test_add_bos_token_false_is_honored(self, spm_processor):
        tok = _build_tokenizer(_spm_meta(spm_processor, **{"tokenizer.ggml.add_bos_token": False}))
        assert tok is not None
        assert tok.encode("Hello") == spm_processor.encode("Hello")

    def test_control_token_text_maps_to_its_id(self, spm_processor):
        tok = _build_tokenizer(_spm_meta(spm_processor))
        assert tok is not None
        assert tok.encode("</s>") == [spm_processor.bos_id(), spm_processor.eos_id()]


# ── byte-level BPE ("gpt2") vocab ─────────────────────────────────────

BOT, EOT = "<|begin_of_text|>", "<|end_of_text|>"


@pytest.fixture(scope="module")
def llama3_style_tokenizer():
    """A small byte-level BPE with Llama-3's pre-tokenizer, decoder, BOS
    template and ignore_merges setting."""
    from tokenizers import Regex, Tokenizer, decoders, pre_tokenizers, trainers
    from tokenizers.models import BPE
    from tokenizers.processors import TemplateProcessing
    from transformers import PreTrainedTokenizerFast

    tok = Tokenizer(BPE())
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(LLAMA3_PATTERN), behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(  # pyright: ignore[reportCallIssue]
        vocab_size=500,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=[BOT, EOT],
        show_progress=False,
    )
    tok.train_from_iterator(CORPUS_LINES, trainer)
    spec = json.loads(tok.to_str())
    spec["model"]["ignore_merges"] = True
    tok = Tokenizer.from_str(json.dumps(spec))
    tok.post_processor = TemplateProcessing(
        single=f"{BOT} $A", pair=f"{BOT} $A {BOT} $B:1",
        special_tokens=[(BOT, tok.token_to_id(BOT))],
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=tok, bos_token=BOT, eos_token=EOT,
        clean_up_tokenization_spaces=False,
    )


def _bpe_meta(ref, **extra) -> dict:
    spec = json.loads(ref.backend_tokenizer.to_str())
    vocab = ref.get_vocab()
    tokens = [""] * len(vocab)
    for t, i in vocab.items():
        tokens[i] = t
    merges = [m if isinstance(m, str) else " ".join(m) for m in spec["model"]["merges"]]
    meta = {
        "tokenizer.ggml.model": "gpt2",
        "tokenizer.ggml.pre": "llama-bpe",
        "tokenizer.ggml.tokens": tokens,
        "tokenizer.ggml.merges": merges,
        "tokenizer.ggml.token_type": [3 if t in (BOT, EOT) else 1 for t in tokens],
        "tokenizer.ggml.bos_token_id": vocab[BOT],
        "tokenizer.ggml.eos_token_id": vocab[EOT],
        "tokenizer.ggml.add_bos_token": True,
    }
    meta.update(extra)
    return meta


class TestByteLevelVocab:
    @pytest.mark.parametrize("text", TEXTS)
    def test_ids_match_reference(self, llama3_style_tokenizer, text):
        tok = _build_tokenizer(_bpe_meta(llama3_style_tokenizer))
        assert tok is not None
        assert tok.encode(text) == llama3_style_tokenizer.encode(text)

    @pytest.mark.parametrize("text", TEXTS)
    def test_decode_roundtrip(self, llama3_style_tokenizer, text):
        tok = _build_tokenizer(_bpe_meta(llama3_style_tokenizer))
        assert tok is not None
        assert tok.decode(tok.encode(text), skip_special_tokens=True) == text

    def test_no_bos_without_add_bos_token(self, llama3_style_tokenizer):
        meta = _bpe_meta(llama3_style_tokenizer)
        del meta["tokenizer.ggml.add_bos_token"]
        tok = _build_tokenizer(meta)
        assert tok is not None
        assert tok.encode("Hello world")[0] != llama3_style_tokenizer.bos_token_id

    def test_gpt2_pre_roundtrips(self, llama3_style_tokenizer):
        tok = _build_tokenizer(_bpe_meta(llama3_style_tokenizer, **{"tokenizer.ggml.pre": "gpt-2"}))
        assert tok is not None
        text = "Hello world, it's 2718"
        assert tok.decode(tok.encode(text), skip_special_tokens=True) == text


# ── writer → reader ───────────────────────────────────────────────────

def _export_with(tokenizer, path):
    from llm_surgeon.gguf_writer import export_hf_to_gguf

    config = LlamaConfig(  # pyright: ignore[reportCallIssue]
        vocab_size=len(tokenizer.get_vocab()), hidden_size=32, intermediate_size=64,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=64,
    )
    return export_hf_to_gguf(LlamaForCausalLM(config), tokenizer, path)


class TestWriterTokenizer:
    def test_llama3_style_export_writes_pre_and_reloads(self, llama3_style_tokenizer, tmp_path):
        pytest.importorskip("gguf")
        out = _export_with(llama3_style_tokenizer, tmp_path / "bpe.gguf")
        with GGUFFile(out) as g:
            assert g.metadata["tokenizer.ggml.model"] == "gpt2"
            assert g.metadata["tokenizer.ggml.pre"] == "llama-bpe"
        _, tok = load_gguf_as_hf(out)
        assert tok is not None
        for text in TEXTS:
            assert tok.encode(text) == llama3_style_tokenizer.encode(text), text

    def test_llama_cpp_tokenizes_export_like_reference(self, llama3_style_tokenizer, tmp_path):
        """llama.cpp's own tokenizer, reading the exported pre name, agrees."""
        pytest.importorskip("gguf")
        llama_cpp = pytest.importorskip("llama_cpp")
        out = _export_with(llama3_style_tokenizer, tmp_path / "bpe.gguf")
        llm = llama_cpp.Llama(model_path=os.fspath(out), vocab_only=True, verbose=False)
        for text in TEXTS:
            ids = llm.tokenize(text.encode(), add_bos=True, special=True)
            assert ids == llama3_style_tokenizer.encode(text), text
