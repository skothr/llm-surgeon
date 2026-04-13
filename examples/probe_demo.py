"""Demo: probe module capabilities on TinyLlama 1.1B.

This script demonstrates the core features of llm_surgeon.probe:
  1. Logit lens — project intermediate hidden states into token space
  2. Hidden state extraction — track residual stream evolution
  3. Intervention — modify hidden states and observe downstream effects
  4. Noise injection — test prediction robustness
  5. Sub-layer comparison — attention vs FFN contributions

All analysis is deterministic (no sampling, no temperature). The logit lens
computes softmax(logits) directly from intermediate hidden states, so the
probabilities shown are the model's raw beliefs at each layer, not sampled
outputs.
"""

import sys
sys.path.insert(0, "/home/ai/ai-projects/llm/testing")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from llm_surgeon.probe import (
    logit_lens, extract_hidden_states, intervene,
    Intervention, ops, layer_predictions_table,
)

# -------------------------------------------------------------------------
# Model configuration
# -------------------------------------------------------------------------

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
CACHE_DIR = "/home/ai/ai-projects/llm/models"

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)

# Load in fp16 with automatic device placement (GPU if available, else CPU).
# Note: `dtype` is the current parameter name (torch_dtype is deprecated).
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, cache_dir=CACHE_DIR, dtype=torch.float16, device_map="auto",
)

# eval() disables dropout (not that TinyLlama uses it, but good practice).
# All forward passes below use torch.no_grad() internally via the probe API.
model.eval()

# Print model details so results can be reproduced and compared.
device = model.model.embed_tokens.weight.device
print(f"Model:       {MODEL_ID}")
print(f"Layers:      {len(model.model.layers)}")
print(f"Hidden size: {model.config.hidden_size}")
print(f"Heads:       {model.config.num_attention_heads}")
print(f"Vocab size:  {model.config.vocab_size}")
print(f"Dtype:       {model.dtype}")
print(f"Device:      {device}")
print(f"Sampling:    none (deterministic logit lens, no temperature)")
print()


def get_final_predictions(predictions, prompt_tokens, sublayer="ffn"):
    """Helper: extract the final layer's predictions for the last position."""
    last_pos = len(prompt_tokens) - 1
    final_layer = max(p["layer"] for p in predictions)
    return [
        p for p in predictions
        if p["layer"] == final_layer
        and p["position"] == last_pos
        and p["sublayer"] == sublayer
    ]


# -------------------------------------------------------------------------
# 1. Logit Lens — watch predictions crystallize layer by layer
# -------------------------------------------------------------------------
# The logit lens projects each layer's hidden state through the model's
# output head (final RMSNorm -> lm_head). This shows what the model would
# predict if it stopped processing at that layer.
#
# For each layer, we capture two points:
#   "attn" — after the attention sublayer's residual add (pre-FFN)
#   "ffn"  — after the FFN sublayer's residual add (full layer output)

print("=" * 60)
print("1. LOGIT LENS — 'The capital of France is'")
print("=" * 60)

prompt = "The capital of France is"

# positions=[-1] means we only analyze the last token position, which is
# where the model's next-token prediction lives.
result = logit_lens(model, tokenizer, prompt, top_k=5, positions=[-1])

# The summary table shows: layer index, sublayer, top-1 token, its
# probability, and the top-3 tokens at each capture point.
print(layer_predictions_table(result, position=-1))

# prediction_flips counts how many times the top-1 prediction changes
# across layers — a rough measure of how much the model "changes its mind".
last_pos = len(result.prompt_tokens) - 1
print(f"\nPrediction flips (last position): {result.prediction_flips(last_pos)}")

# first_correct_layer finds the earliest layer where a target token
# appears as the top-1 prediction. Useful for factual recall analysis.
first = result.first_correct_layer(position=last_pos, target_token=" Paris")
if first is not None:
    print(f"' Paris' first appears as top-1 at layer {first}")
else:
    print("' Paris' never appears as top-1 (try checking top-5)")


# -------------------------------------------------------------------------
# 2. Hidden state evolution — cosine similarity between layers
# -------------------------------------------------------------------------
# Extract the raw residual stream vectors at each layer boundary (post-FFN).
# Cosine similarity between consecutive layers shows how much each layer
# changes the representation:
#   ~1.0 = layer barely modifies the hidden state
#   ~0.5 = major transformation
#
# This complements the logit lens: you can see WHERE the representation
# changes most, even when the top-1 prediction doesn't flip.

print("\n" + "=" * 60)
print("2. HIDDEN STATE EVOLUTION — cosine similarity (last position)")
print("=" * 60)

# sublayers=("ffn",) captures only post-FFN states (one per layer).
# Use sublayers=("attn", "ffn") for both capture points.
hs = extract_hidden_states(model, tokenizer, prompt, sublayers=("ffn",))
print(f"Captured {len(hs.states)} layer states, each shape: (seq_len, {model.config.hidden_size})")

# Compare consecutive layers at the last token position.
prev_key = None
for key in sorted(hs.states.keys()):
    if prev_key is not None:
        sim = hs.cosine_similarity(prev_key, key, position=-1)
        print(f"  Layer {prev_key[0]:>2} -> {key[0]:>2}: cosine_sim = {sim:.4f}")
    prev_key = key


# -------------------------------------------------------------------------
# 3. Intervention — what happens when we zero out a layer's FFN?
# -------------------------------------------------------------------------
# intervene() modifies hidden states during the forward pass using PyTorch
# hooks. Here we scale layer 10's FFN output to zero, effectively removing
# that layer's FFN contribution from the residual stream.
#
# With capture_logit_lens=True, we also get logit lens data from the
# modified forward pass, so we can see how the intervention propagates.

print("\n" + "=" * 60)
print("3. INTERVENTION — zero FFN at layer 10")
print("=" * 60)

# Baseline: normal forward pass with logit lens.
baseline = logit_lens(model, tokenizer, prompt, top_k=3, positions=[-1])

# Intervention: scale layer 10 FFN output by 0.0 (zeroing it out).
# ops.scale(0.0) returns a callable that multiplies the hidden state by 0.
modified = intervene(
    model, tokenizer, prompt,
    interventions=[Intervention(layer=10, sublayer="ffn", fn=ops.scale(0.0))],
    capture_logit_lens=True,
    top_k=3,
)

print(f"\nBaseline top-3 (last position, final layer):")
for t in get_final_predictions(baseline.predictions, baseline.prompt_tokens)[0]["top_k"]:
    print(f"  {t['token']:>15} ({t['prob']:.3f})")

print(f"\nWith layer 10 FFN zeroed:")
mod_final = get_final_predictions(
    modified.logit_lens_result.predictions,
    modified.logit_lens_result.prompt_tokens,
)
if mod_final:
    for t in mod_final[0]["top_k"]:
        print(f"  {t['token']:>15} ({t['prob']:.3f})")


# -------------------------------------------------------------------------
# 4. Noise injection — test prediction robustness
# -------------------------------------------------------------------------
# Inject Gaussian noise at layer 5's FFN output with increasing standard
# deviation. This tests how robust the model's factual recall is to
# perturbation at an early-middle layer.
#
# seed=42 ensures the noise is deterministic across runs, so results are
# reproducible. Each noise level gets its own seed-42 noise pattern.

print("\n" + "=" * 60)
print("4. NOISE INJECTION — increasing noise at layer 5 FFN")
print("=" * 60)

for std in [0.0, 0.5, 1.0, 2.0, 5.0]:
    # ops.noise(std, seed=42) adds N(0, std) noise to the hidden state.
    r = intervene(
        model, tokenizer, prompt,
        interventions=[Intervention(layer=5, sublayer="ffn", fn=ops.noise(std, seed=42))],
        capture_logit_lens=True,
        top_k=1,
    )
    final = get_final_predictions(
        r.logit_lens_result.predictions,
        r.logit_lens_result.prompt_tokens,
    )
    if final:
        top = final[0]["top_k"][0]
        print(f"  noise std={std:<4}  ->  {top['token']:>15} ({top['prob']:.3f})")


# -------------------------------------------------------------------------
# 5. Sub-layer comparison — attention vs FFN contribution at each layer
# -------------------------------------------------------------------------
# Show the top-1 prediction at both the post-attention and post-FFN capture
# points. This reveals which sublayer drives each prediction shift.
#
# Common pattern: attention surfaces relevant context (e.g., "France"),
# then the FFN retrieves the associated fact (e.g., "Paris").

print("\n" + "=" * 60)
print("5. ATTENTION vs FFN — top-1 prediction at each sublayer")
print("=" * 60)

result = logit_lens(model, tokenizer, prompt, top_k=1, positions=[-1])
for p in result.predictions:
    if p["position"] == last_pos and p["top_k"]:
        top = p["top_k"][0]
        print(f"  Layer {p['layer']:>2} {p['sublayer']:>4}:  {top['token']:>15} ({top['prob']:.3f})")

print("\nDone.")
