#!/usr/bin/env python3
"""Capture authentic SmolLM2/Llama Q/K/V using a small NumPy FP32 reference.

The checkpoint values are unchanged. BF16 checkpoint tensors are represented
exactly in FP32; all arithmetic here is FP32. This is a numerical reference,
not an inference-speed baseline. Q and K are saved *after* RoPE, V before GQA
replication. A Transformers comparison is available with --validate-transformers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import struct
import sys
import tempfile
import time
from typing import Callable, Iterable

import numpy as np

MODEL_ID = "HuggingFaceTB/SmolLM2-135M"
MODEL_REVISION = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


class SafeTensorWeights:
    """Minimal read-only safetensors loader for numeric BF16/F16/F32 weights."""

    def __init__(self, path: str | Path, cache: bool = True):
        self.path = Path(path)
        with self.path.open("rb") as f:
            header_len = struct.unpack("<Q", f.read(8))[0]
            if header_len > 100_000_000:
                raise ValueError("Unreasonable safetensors header length")
            self.header = json.loads(f.read(header_len))
        self.data_start = 8 + header_len
        self.cache_enabled = cache
        self.cache: dict[str, np.ndarray] = {}

    def __getitem__(self, key: str) -> np.ndarray:
        if key in self.cache:
            return self.cache[key]
        info = self.header[key]
        dtype_name = info["dtype"]
        dtype = {"BF16": "<u2", "F16": "<f2", "F32": "<f4"}.get(dtype_name)
        if dtype is None:
            raise ValueError(f"Unsupported checkpoint dtype {dtype_name}")
        shape = tuple(info["shape"])
        start, end = info["data_offsets"]
        size = math.prod(shape) * np.dtype(dtype).itemsize
        if start < 0 or end - start != size or self.data_start + end > self.path.stat().st_size:
            raise ValueError(f"Invalid safetensors span for {key}")
        raw = np.memmap(self.path, mode="r", dtype=dtype, offset=self.data_start + start, shape=shape)
        if dtype_name == "BF16":
            result = (np.asarray(raw, dtype=np.uint32) << np.uint32(16)).view(np.float32)
        else:
            result = np.asarray(raw, dtype=np.float32)
        if self.cache_enabled:
            self.cache[key] = result
        return result


def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    return x * np.reciprocal(np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + np.float32(eps))) * weight


def rope(x: np.ndarray, positions: np.ndarray, theta: float) -> np.ndarray:
    """Llama's non-interleaved half-rotation, x=[heads,tokens,head_dim]."""
    dim = x.shape[-1]
    if dim % 2:
        raise ValueError("RoPE requires even head_dim")
    inv_freq = np.float32(1.0) / np.power(np.float32(theta), np.arange(0, dim, 2, dtype=np.float32) / np.float32(dim))
    freq = np.asarray(positions, np.float32)[:, None] * inv_freq[None, :]
    cos = np.concatenate((np.cos(freq), np.cos(freq)), axis=-1)[None]
    sin = np.concatenate((np.sin(freq), np.sin(freq)), axis=-1)[None]
    rotated = np.concatenate((-x[..., dim // 2 :], x[..., : dim // 2]), axis=-1)
    return x * cos + rotated * sin


def causal_attention(q: np.ndarray, k: np.ndarray, v: np.ndarray,
                     query_positions: np.ndarray | None = None) -> np.ndarray:
    """FP32 causal GQA, without keeping replicated K/V arrays."""
    hq, tq, d = q.shape
    hk, tk, kd = k.shape
    if v.shape != k.shape or d != kd or hq % hk:
        raise ValueError("Incompatible Q/K/V shapes")
    if query_positions is None:
        query_positions = np.arange(tk - tq, tk)
    queries = q.reshape(hk, hq // hk, tq, d)
    scores = np.matmul(queries, k[:, None].swapaxes(-1, -2)) * np.float32(d**-0.5)
    mask = np.arange(tk)[None, :] <= np.asarray(query_positions)[:, None]
    scores = np.where(mask[None, None], scores, np.float32(-np.inf))
    scores -= np.max(scores, axis=-1, keepdims=True)
    probs = np.exp(scores)
    probs /= np.sum(probs, axis=-1, keepdims=True)
    return np.matmul(probs, v[:, None]).reshape(hq, tq, d)


class NumpyLlama:
    """Transparent full-sequence Llama, restricted to the selected architecture."""

    def __init__(self, model_dir: str | Path, cache_weights: bool = True):
        self.model_dir = Path(model_dir)
        self.config = json.loads((self.model_dir / "config.json").read_text())
        cfg = self.config
        unsupported = []
        if cfg.get("model_type") != "llama":
            unsupported.append("non-Llama model")
        if cfg.get("rope_scaling"):
            unsupported.append("scaled RoPE")
        if cfg.get("rope_interleaved", False):
            unsupported.append("interleaved RoPE")
        if cfg.get("attention_bias", False) or cfg.get("mlp_bias", False):
            unsupported.append("projection biases")
        if cfg.get("hidden_act", "silu") != "silu":
            unsupported.append("non-SiLU MLP")
        if cfg.get("sliding_window"):
            unsupported.append("sliding-window attention")
        if unsupported:
            raise ValueError("Unsupported reference configuration: " + ", ".join(unsupported))
        self.weights = SafeTensorWeights(self.model_dir / "model.safetensors", cache=cache_weights)
        self.hq = int(cfg["num_attention_heads"])
        self.hkv = int(cfg["num_key_value_heads"])
        self.dim = int(cfg["hidden_size"])
        self.head_dim = self.dim // self.hq
        if self.dim % self.hq or self.hq % self.hkv:
            raise ValueError("Invalid head configuration")
        self.num_layers = int(cfg["num_hidden_layers"])

    def forward(self, token_ids: np.ndarray | list[int],
                capture_layers: Iterable[int] | None = None,
                kv_transform: Callable[[int, np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]] | None = None,
                capture_pre_rope: bool = False, verbose: bool = False) -> dict[str, np.ndarray]:
        """Evaluate a single causal sequence, optionally replacing K/V per layer.

        kv_transform sees [H,T,D] tensors after RoPE. For a causal model-quality
        experiment it MUST NOT derive earlier cache metadata from future tokens.
        Captures contain the actual K/V used by this forward pass.
        """
        ids = np.asarray(token_ids, dtype=np.int64)
        if ids.ndim != 1 or ids.size < 1 or ids.size > int(self.config["max_position_embeddings"]):
            raise ValueError("Expected 1..max_position_embeddings tokens")
        if ids.min() < 0 or ids.max() >= int(self.config["vocab_size"]):
            raise ValueError("Out-of-vocabulary token ID")
        layers = list(range(self.num_layers)) if capture_layers is None else sorted(set(capture_layers))
        if any(i < 0 or i >= self.num_layers for i in layers):
            raise ValueError("Invalid capture layer")
        capture_set = set(layers)
        captures: dict[str, list[np.ndarray]] = {n: [] for n in ("q", "k", "v")}
        if capture_pre_rope:
            captures.update(q_pre_rope=[], k_pre_rope=[])
        positions = np.arange(ids.size, dtype=np.int64)
        w = self.weights
        x = w["model.embed_tokens.weight"][ids].copy()
        eps = float(self.config["rms_norm_eps"])
        for layer in range(self.num_layers):
            prefix = f"model.layers.{layer}."
            h = rms_norm(x, w[prefix + "input_layernorm.weight"], eps)
            q0 = (h @ w[prefix + "self_attn.q_proj.weight"].T).reshape(ids.size, self.hq, self.head_dim).transpose(1, 0, 2)
            k0 = (h @ w[prefix + "self_attn.k_proj.weight"].T).reshape(ids.size, self.hkv, self.head_dim).transpose(1, 0, 2)
            v = (h @ w[prefix + "self_attn.v_proj.weight"].T).reshape(ids.size, self.hkv, self.head_dim).transpose(1, 0, 2)
            q = rope(q0, positions, float(self.config["rope_theta"]))
            k = rope(k0, positions, float(self.config["rope_theta"]))
            if kv_transform is not None:
                k_new, v_new = kv_transform(layer, q, k, v)
                if k_new.shape != k.shape or v_new.shape != v.shape:
                    raise ValueError("kv_transform changed tensor shapes")
                k, v = np.asarray(k_new, np.float32), np.asarray(v_new, np.float32)
            if layer in capture_set:
                for name, value in (("q", q), ("k", k), ("v", v)):
                    captures[name].append(np.ascontiguousarray(value))
                if capture_pre_rope:
                    captures["q_pre_rope"].append(np.ascontiguousarray(q0))
                    captures["k_pre_rope"].append(np.ascontiguousarray(k0))
            attn = causal_attention(q, k, v).transpose(1, 0, 2).reshape(ids.size, self.dim)
            x = x + attn @ w[prefix + "self_attn.o_proj.weight"].T
            h = rms_norm(x, w[prefix + "post_attention_layernorm.weight"], eps)
            gate = h @ w[prefix + "mlp.gate_proj.weight"].T
            up = h @ w[prefix + "mlp.up_proj.weight"].T
            with np.errstate(over="ignore"):
                activated = gate / (np.float32(1) + np.exp(-gate))
            x = x + (activated * up) @ w[prefix + "mlp.down_proj.weight"].T
            if verbose:
                print(f"captured layer {layer + 1}/{self.num_layers}", file=sys.stderr, flush=True)
        final_hidden = rms_norm(x, w["model.norm.weight"], eps)
        out = {name: np.stack(vals) for name, vals in captures.items() if vals}
        out.update(final_hidden=np.asarray(final_hidden, np.float32), token_ids=ids,
                   layer_ids=np.asarray(layers, np.int32), positions=positions)
        return out

    def logits(self, hidden: np.ndarray) -> np.ndarray:
        key = "model.embed_tokens.weight" if self.config.get("tie_word_embeddings", False) else "lm_head.weight"
        return hidden @ self.weights[key].T

    def prefill(self, token_ids: np.ndarray | list[int]) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
        """Return next-token logits [V] and one (K,V) pair per layer.

        This prefill is uncompressed. The caller can then encode completed
        historical pages in each returned cache without using future tokens.
        """
        result = self.forward(token_ids)
        caches = [(result["k"][layer].copy(), result["v"][layer].copy()) for layer in range(self.num_layers)]
        return self.logits(result["final_hidden"][-1:])[0], caches

    def step(self, token: int, position: int,
             caches: list[tuple[np.ndarray, np.ndarray]],
             kv_transform: Callable[[int, np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]] | None = None,
             ) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
        """One real incremental token; returns logits and new per-layer caches.

        Previous caches are not mutated. They may already contain the numerical
        reconstructions of compressed pages. Newly computed K/V then reflect the
        perturbed earlier layers and history, so error feedback is retained.
        Position is absolute from zero and each old cache must have that length.
        This CPU reference stores reconstructed arrays, not a compressed runtime.
        """
        if len(caches) != self.num_layers or position < 0 or position >= self.config["max_position_embeddings"]:
            raise ValueError("Invalid layer cache count or token position")
        if token < 0 or token >= self.config["vocab_size"]:
            raise ValueError("Out-of-vocabulary token ID")
        expected = (self.hkv, position, self.head_dim)
        if any(k.shape != expected or v.shape != expected for k, v in caches):
            raise ValueError(f"All cache tensors must have shape {expected}")
        w = self.weights
        x = w["model.embed_tokens.weight"][[token]].copy()
        eps = float(self.config["rms_norm_eps"])
        new_caches = []
        positions = np.asarray([position], np.int64)
        for layer in range(self.num_layers):
            prefix = f"model.layers.{layer}."
            h = rms_norm(x, w[prefix + "input_layernorm.weight"], eps)
            q0 = (h @ w[prefix + "self_attn.q_proj.weight"].T).reshape(1, self.hq, self.head_dim).transpose(1, 0, 2)
            k0 = (h @ w[prefix + "self_attn.k_proj.weight"].T).reshape(1, self.hkv, self.head_dim).transpose(1, 0, 2)
            v0 = (h @ w[prefix + "self_attn.v_proj.weight"].T).reshape(1, self.hkv, self.head_dim).transpose(1, 0, 2)
            q = rope(q0, positions, float(self.config["rope_theta"]))
            knew = rope(k0, positions, float(self.config["rope_theta"]))
            oldk, oldv = caches[layer]
            k = np.concatenate((oldk, knew), axis=1)
            v = np.concatenate((oldv, v0), axis=1)
            if kv_transform is not None:
                k, v = kv_transform(layer, q, k, v)
                k, v = np.asarray(k, np.float32), np.asarray(v, np.float32)
                new_shape = (self.hkv, position + 1, self.head_dim)
                if k.shape != new_shape or v.shape != new_shape:
                    raise ValueError("kv_transform changed cache shape")
            new_caches.append((k, v))
            attn = causal_attention(q, k, v, query_positions=positions).transpose(1, 0, 2).reshape(1, self.dim)
            x = x + attn @ w[prefix + "self_attn.o_proj.weight"].T
            h = rms_norm(x, w[prefix + "post_attention_layernorm.weight"], eps)
            gate = h @ w[prefix + "mlp.gate_proj.weight"].T
            up = h @ w[prefix + "mlp.up_proj.weight"].T
            with np.errstate(over="ignore"):
                activated = gate / (np.float32(1) + np.exp(-gate))
            x = x + (activated * up) @ w[prefix + "mlp.down_proj.weight"].T
        final_hidden = rms_norm(x, w["model.norm.weight"], eps)
        return self.logits(final_hidden)[0], new_caches

    def validate_incremental(self, token_ids: np.ndarray | list[int], max_tokens: int = 32) -> dict:
        ids = np.asarray(token_ids, np.int64)[:max_tokens]
        if len(ids) < 2:
            raise ValueError("Incremental validation requires at least two tokens")
        full = self.forward(ids)
        _, cache = self.prefill(ids[:-1])
        logits, cache = self.step(int(ids[-1]), len(ids) - 1, cache)
        metrics = {"logits": error_stats(logits, self.logits(full["final_hidden"][-1:])[0]),
                   "k": error_stats(np.stack([kv[0] for kv in cache]), full["k"]),
                   "v": error_stats(np.stack([kv[1] for kv in cache]), full["v"])}
        passed = all(s["max_abs"] < 1e-3 and s["relative_l2"] < 1e-4 for s in metrics.values())
        return {"status": "passed" if passed else "failed", "tokens": len(ids), "metrics": metrics,
                "comparison": "full causal forward vs prefill(T-1) plus step(T-1)"}

    def nll(self, hidden: np.ndarray, token_ids: np.ndarray, chunk: int = 64) -> np.ndarray:
        """Unreduced next-token NLL; position i predicts token i+1."""
        losses = []
        for start in range(0, len(token_ids) - 1, chunk):
            end = min(start + chunk, len(token_ids) - 1)
            z = self.logits(hidden[start:end])
            m = z.max(-1)
            lse = m + np.log(np.exp(z - m[:, None]).sum(-1))
            target = np.asarray(token_ids[start + 1 : end + 1])
            losses.append(lse - z[np.arange(end - start), target])
        return np.concatenate(losses) if losses else np.empty(0, np.float32)


def tokenize(model_dir: Path, text: str, max_tokens: int) -> np.ndarray:
    try:
        from tokenizers import Tokenizer
    except ImportError as exc:
        raise RuntimeError("Install tokenizers or supply --token-ids-file") from exc
    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    # The checkpoint uses a GPT2 tokenizer. No implicit BOS or chat template.
    ids = tokenizer.encode(text, add_special_tokens=False).ids[:max_tokens]
    if not ids:
        raise ValueError("Text produced no tokens")
    return np.asarray(ids, dtype=np.int64)


def error_stats(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    delta = np.asarray(a, np.float64) - np.asarray(b, np.float64)
    norm = np.linalg.norm(np.asarray(b, np.float64).ravel())
    return {"max_abs": float(np.abs(delta).max()), "rms": float(np.sqrt(np.mean(delta * delta))),
            "relative_l2": float(np.linalg.norm(delta.ravel()) / max(norm, 1e-30))}


def validate_transformers(model: NumpyLlama, token_ids: np.ndarray, max_tokens: int = 32) -> dict:
    """Independent HF eager forward; actual HF post-RoPE tensors are intercepted."""
    import torch
    import transformers
    from transformers import AutoModelForCausalLM
    from transformers.models.llama import modeling_llama

    ids = np.asarray(token_ids[:max_tokens], np.int64)
    ref = model.forward(ids)
    reference = AutoModelForCausalLM.from_pretrained(
        str(model.model_dir), local_files_only=True, torch_dtype=torch.float32,
        attn_implementation="eager",
    ).eval()
    torch.set_num_threads(min(int(os.environ.get("ASTCKV_CPU_THREADS", "4")), os.cpu_count() or 1))
    captured_q, captured_k, captured_v = [], [], []
    original = modeling_llama.apply_rotary_pos_emb

    def intercept_rope(*args, **kwargs):
        qo, ko = original(*args, **kwargs)
        captured_q.append(qo.detach().cpu().numpy()[0].copy())
        captured_k.append(ko.detach().cpu().numpy()[0].copy())
        return qo, ko

    hooks = []
    for layer in reference.model.layers:
        def save_v(_module, _inputs, output):
            captured_v.append(output.detach().cpu().numpy()[0].reshape(len(ids), model.hkv, model.head_dim).transpose(1, 0, 2).copy())
        hooks.append(layer.self_attn.v_proj.register_forward_hook(save_v))
    try:
        modeling_llama.apply_rotary_pos_emb = intercept_rope
        with torch.inference_mode():
            result = reference(torch.tensor(ids)[None], use_cache=False, output_hidden_states=True)
    finally:
        modeling_llama.apply_rotary_pos_emb = original
        for hook in hooks:
            hook.remove()
    if len(captured_q) != model.num_layers:
        raise RuntimeError("HF execution did not expose one RoPE operation per layer")
    metrics = {"q": error_stats(ref["q"], np.stack(captured_q)),
               "k": error_stats(ref["k"], np.stack(captured_k)),
               "v": error_stats(ref["v"], np.stack(captured_v)),
               "final_hidden": error_stats(ref["final_hidden"], result.hidden_states[-1][0].numpy()),
               "logits": error_stats(model.logits(ref["final_hidden"]), result.logits[0].numpy())}
    passed = all(s["max_abs"] < 1e-3 and s["relative_l2"] < 1e-4 for s in metrics.values())
    return {"status": "passed" if passed else "failed", "tokens": len(ids), "metrics": metrics,
            "torch_version": torch.__version__, "transformers_version": transformers.__version__,
            "dtype": "FP32", "attention": "eager", "thresholds": {"max_abs": 1e-3, "relative_l2": 1e-4}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--text-file", type=Path)
    src.add_argument("--text")
    src.add_argument("--token-ids-file", type=Path, help="JSON array of integer token IDs")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--layers", default="all")
    parser.add_argument("--include-pre-rope", action="store_true")
    parser.add_argument("--save-logits", action="store_true")
    parser.add_argument("--validate-transformers", action="store_true")
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--split", choices=["calibration", "tune", "development", "pilot_holdout", "confirmation", "smoke"], default="development")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    text = args.text_file.read_text() if args.text_file else args.text
    if args.token_ids_file:
        ids = np.asarray(json.loads(args.token_ids_file.read_text()), np.int64)[:args.max_tokens]
    else:
        ids = tokenize(args.model_dir, text, args.max_tokens)
    model = NumpyLlama(args.model_dir)
    layers = None if args.layers == "all" else [int(i) for i in args.layers.split(",")]
    started = time.perf_counter()
    arrays = model.forward(ids, capture_layers=layers, capture_pre_rope=args.include_pre_rope, verbose=args.verbose)
    arrays["nll"] = model.nll(arrays["final_hidden"], ids)
    arrays["last_logits"] = model.logits(arrays["final_hidden"][-8:])
    if args.save_logits:
        arrays["logits"] = model.logits(arrays["final_hidden"])
    numerical_wall_seconds = time.perf_counter() - started
    validation = validate_transformers(model, ids) if args.validate_transformers else {"status": "not_run"}
    if validation["status"] == "failed":
        raise RuntimeError("Independent reference comparison failed: " + json.dumps(validation))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Publish only a complete readable archive. Downstream codec workers may
    # otherwise observe a ZIP header before its central directory is written.
    with tempfile.NamedTemporaryFile(dir=args.out.parent, prefix=args.out.stem + ".", suffix=".npz", delete=False) as temporary:
        temporary_path = Path(temporary.name)
        np.savez_compressed(temporary, **arrays)
    try:
        with np.load(temporary_path, allow_pickle=False) as check:
            for name in check.files:
                if check[name].shape != arrays[name].shape:
                    raise RuntimeError(f"Capture shape verification failed for {name}")
        os.replace(temporary_path, args.out)
    finally:
        temporary_path.unlink(missing_ok=True)
    metadata = {
        "schema": "astckv.qkv_capture.v1", "model_id": args.model_id, "model_revision": args.model_revision,
        "weights_file_sha256": sha256(args.model_dir / "model.safetensors"),
        "config_file_sha256": sha256(args.model_dir / "config.json"), "config": model.config,
        "tokenizer_file_sha256": sha256(args.model_dir / "tokenizer.json") if (args.model_dir / "tokenizer.json").exists() else None,
        "capture_file_sha256": sha256(args.out), "source_text": text,
        "source_text_sha256": hashlib.sha256(text.encode()).hexdigest() if text is not None else None,
        "tokenization": "checkpoint tokenizer.json; add_special_tokens=False; no chat template",
        "split": args.split, "token_count": len(ids), "layer_ids": arrays["layer_ids"].tolist(),
        "layout": {"q": "[captured_layers,query_heads,tokens,head_dim]", "k": "[captured_layers,kv_heads,tokens,head_dim]", "v": "[captured_layers,kv_heads,tokens,head_dim]"},
        "qk_position": "post_RoPE", "v_position": "projected_values_before_GQA_replication",
        "batch": 1, "causal": True, "weights_original_dtype": "BF16", "arithmetic_dtype": "FP32",
        "weight_quantization": "none beyond original BF16 checkpoint", "backend": "NumPy CPU reference",
        "numpy_version": np.__version__, "python_version": sys.version, "platform": platform.platform(),
        "baseline_mean_nll": float(arrays["nll"].mean()) if arrays["nll"].size else None,
        "baseline_perplexity": float(np.exp(arrays["nll"].astype(np.float64).mean())) if arrays["nll"].size else None,
        "numerical_wall_seconds": numerical_wall_seconds,
        "timing_warning": "CPU reference wall time is not a Metal or inference performance result",
        "reference_validation": validation,
    }
    args.out.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "shapes": {k: list(v.shape) for k, v in arrays.items()},
                      "mean_nll": metadata["baseline_mean_nll"], "perplexity": metadata["baseline_perplexity"],
                      "wall_seconds": numerical_wall_seconds, "reference_validation": validation}, indent=2))


if __name__ == "__main__":
    main()
