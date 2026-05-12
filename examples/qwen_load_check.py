"""Pre-flight: confirm Qwen/Qwen2.5-7B-Instruct loads in nf4 on this box.

Smallest base model with a released Anthropic NLA checkpoint
(kitft/nla-models, NLA paper 2026-05-07). ~15 GB on disk; ~4-5 GB VRAM
after nf4 quantization.
"""

import os
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")

import sys
from io import TextIOWrapper
from typing import cast
cast(TextIOWrapper, sys.stdout).reconfigure(line_buffering=True)

import shutil
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from llm_surgeon import surgery  # noqa: F401 — kept for cache_dir parity


MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
NLA_TARGET_LAYER = 20  # kitft/nla-models pins layer 20/28 for this base.


def _gb(n_bytes: int) -> float:
    return n_bytes / (1024**3)


def _disk_free_gb(path: str = ".") -> float:
    return _gb(shutil.disk_usage(path).free)


def _gpu_state() -> tuple[float, float] | None:
    if not torch.cuda.is_available():
        return None
    free, total = cast(tuple[int, int], torch.cuda.mem_get_info())
    return _gb(total - free), _gb(total)


def main() -> None:
    print(f"disk free before: {_disk_free_gb():.1f} GB")
    gpu = _gpu_state()
    if gpu is not None:
        used, total = gpu
        print(f"gpu before:       {used:.2f} / {total:.2f} GB used")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        cache_dir=surgery.MODEL_CACHE_DIR,
        local_files_only=True,
        use_safetensors=True,
        quantization_config=bnb_config,
        device_map={"": 0},
    )
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, cache_dir=surgery.MODEL_CACHE_DIR, local_files_only=True
    )
    print(f"load time:        {time.time() - t0:.1f} s")

    print(f"disk free after:  {_disk_free_gb():.1f} GB")
    gpu = _gpu_state()
    if gpu is not None:
        used, total = gpu
        print(f"gpu after:        {used:.2f} / {total:.2f} GB used")

    cfg = model.config
    print(
        f"model:            {MODEL_ID}\n"
        f"  num_hidden_layers: {cfg.num_hidden_layers}\n"
        f"  hidden_size:       {cfg.hidden_size}\n"
        f"  vocab_size:        {cfg.vocab_size}\n"
        f"  nla target layer:  {NLA_TARGET_LAYER} (kitft pins layer 20/28)"
    )

    prompt = "The capital of France is"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda:0")
    t0 = time.time()
    with torch.no_grad():
        out = model(input_ids=input_ids, output_hidden_states=True, use_cache=False)
    print(f"forward pass:     {time.time() - t0:.2f} s")

    h20 = out.hidden_states[NLA_TARGET_LAYER]
    print(
        f"hidden_states[{NLA_TARGET_LAYER}]: shape={tuple(h20.shape)}, "
        f"dtype={h20.dtype}, ||h||_2 mean={h20.norm(dim=-1).mean().item():.3f}"
    )

    next_tok = out.logits[0, -1].argmax().item()
    print(f"next token argmax: {next_tok} -> {tokenizer.decode([next_tok])!r}")

    gpu = _gpu_state()
    if gpu is not None:
        used, total = gpu
        print(f"gpu after fwd:    {used:.2f} / {total:.2f} GB used")


if __name__ == "__main__":
    main()
