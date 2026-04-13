"""Quickstart: load a model and run a logit lens probe."""

from llm_surgeon import surgery, probe

model, tokenizer = surgery.load_model(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0", mode="inspect"
)
result = probe.logit_lens(model, tokenizer, "The capital of France is", top_k=5)
print(result.summary())
