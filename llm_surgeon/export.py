"""Export pipeline: HF checkpoint → GGUF → Ollama registration."""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

# Default path to llama.cpp; overridable via env var.
_DEFAULT_LLAMA_CPP_PATH = str(Path(__file__).resolve().parents[2] / "lib" / "llama.cpp")


def save_checkpoint(model, output_dir: str, tokenizer=None) -> str:
    """Save a HuggingFace checkpoint to output_dir.

    Verifies that the saved config.json reflects the actual number of layers.
    Optionally saves a tokenizer alongside the model.

    Returns output_dir.
    Raises ValueError if config.num_hidden_layers doesn't match actual layers.
    """
    actual_layers = len(model.model.layers)
    config_layers = model.config.num_hidden_layers

    if actual_layers != config_layers:
        raise ValueError(
            f"num_hidden_layers mismatch before save: "
            f"actual={actual_layers}, config={config_layers}"
        )

    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)

    if tokenizer is not None:
        tokenizer.save_pretrained(output_dir)

    # Verify the written config is consistent
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path) as f:
        saved_cfg = json.load(f)

    saved_layers = saved_cfg.get("num_hidden_layers")
    if saved_layers != actual_layers:
        raise ValueError(
            f"num_hidden_layers mismatch in saved config.json: "
            f"expected={actual_layers}, found={saved_layers}"
        )

    return output_dir


def to_gguf(
    checkpoint_path: str,
    output_dir: str,
    quantization: Optional[str] = "Q4_K_M",
    llama_cpp_path: Optional[str] = None,
) -> str:
    """Convert an HF checkpoint to GGUF format.

    Steps:
      1. Run convert_hf_to_gguf.py to produce an f16 GGUF.
      2. If quantization is not None, quantize it and remove the f16 intermediate.

    Args:
        checkpoint_path: Path to the saved HF checkpoint directory.
        output_dir: Directory where the GGUF file(s) will be placed.
        quantization: GGUF quantization type (e.g. "Q4_K_M") or None for f16 only.
        llama_cpp_path: Override the llama.cpp installation directory.
                        Defaults to the LLAMA_CPP_PATH env var, then the built-in default.

    Returns:
        Absolute path to the final GGUF file.

    Raises:
        FileNotFoundError: If required llama.cpp tools are not found.
        RuntimeError: If conversion or quantization fails.
    """
    if llama_cpp_path is None:
        llama_cpp_path = os.environ.get("LLAMA_CPP_PATH", _DEFAULT_LLAMA_CPP_PATH)

    convert_script = os.path.join(llama_cpp_path, "convert_hf_to_gguf.py")
    quantize_bin = os.path.join(llama_cpp_path, "build", "bin", "llama-quantize")

    if not os.path.exists(convert_script):
        raise FileNotFoundError(f"convert_hf_to_gguf.py not found at: {convert_script}")

    if quantization is not None and not os.path.exists(quantize_bin):
        raise FileNotFoundError(f"llama-quantize not found at: {quantize_bin}")

    os.makedirs(output_dir, exist_ok=True)

    # Derive a base name from the checkpoint directory
    model_name = Path(checkpoint_path).name
    f16_gguf = os.path.join(output_dir, f"{model_name}-F16.gguf")

    # Step 1: convert to f16 GGUF
    convert_cmd = [
        sys.executable,
        convert_script,
        checkpoint_path,
        "--outfile", f16_gguf,
        "--outtype", "f16",
    ]
    result = subprocess.run(convert_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"GGUF conversion failed (exit {result.returncode}):\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )

    if quantization is None:
        return os.path.abspath(f16_gguf)

    # Step 2: quantize
    quantized_gguf = os.path.join(output_dir, f"{model_name}-{quantization}.gguf")
    quantize_cmd = [quantize_bin, f16_gguf, quantized_gguf, quantization]
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = os.path.dirname(quantize_bin) + ":" + env.get("LD_LIBRARY_PATH", "")
    result = subprocess.run(quantize_cmd, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        raise RuntimeError(
            f"GGUF quantization failed (exit {result.returncode}):\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )

    # Clean up intermediate f16 file
    if os.path.exists(f16_gguf):
        os.remove(f16_gguf)

    return os.path.abspath(quantized_gguf)


def _generate_modelfile(gguf_path: str) -> str:
    """Return Modelfile content for the given GGUF path.

    Uses an absolute path so Ollama can find the model regardless of cwd.
    """
    abs_path = os.path.abspath(gguf_path)
    return f"FROM {abs_path}\n"


def _verify_ollama_registration(name: str) -> bool:
    """Check that a model named `name` appears in Ollama's model list."""
    try:
        import requests
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        if r.status_code != 200:
            return False
        models = r.json().get("models", [])
        return any(name in m.get("name", "") for m in models)
    except Exception:
        return False


def register_ollama(gguf_path: str, name: str) -> None:
    """Register a GGUF model with Ollama.

    Generates a Modelfile, writes it alongside the GGUF, then runs
    `ollama create <name> -f <modelfile>`.

    Raises:
        FileNotFoundError: If gguf_path does not exist.
        RuntimeError: If `ollama create` exits with a non-zero status.
    """
    if not os.path.exists(gguf_path):
        raise FileNotFoundError(f"GGUF file not found: {gguf_path}")

    modelfile_content = _generate_modelfile(gguf_path)

    # Write the Modelfile next to the GGUF
    modelfile_path = os.path.join(os.path.dirname(os.path.abspath(gguf_path)), "Modelfile")
    with open(modelfile_path, "w") as f:
        f.write(modelfile_content)

    cmd = ["ollama", "create", name, "-f", modelfile_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ollama create failed (exit {result.returncode}):\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )

    if not _verify_ollama_registration(name):
        raise RuntimeError(
            f"ollama create exited 0 but model '{name}' is not listed by "
            f"`ollama list` — registration did not take effect."
        )


def full_pipeline(
    model,
    name: str,
    quantization: Optional[str],
    output_dir: str,
    tokenizer=None,
    register: bool = True,
) -> dict:
    """Run the complete export pipeline.

    Steps:
      1. save_checkpoint  → <output_dir>/<name>/checkpoint/
      2. to_gguf          → <output_dir>/<name>/gguf/
      3. register_ollama  → optional

    Returns:
        dict with keys: checkpoint_path, gguf_path, registered
    """
    checkpoint_path = os.path.join(output_dir, name, "checkpoint")
    gguf_dir = os.path.join(output_dir, name, "gguf")

    save_checkpoint(model, checkpoint_path, tokenizer=tokenizer)
    gguf_path = to_gguf(checkpoint_path, gguf_dir, quantization=quantization)

    registered = False
    if register:
        register_ollama(gguf_path, name)
        registered = True

    return {
        "checkpoint_path": checkpoint_path,
        "gguf_path": gguf_path,
        "registered": registered,
    }
