"""Real astcenc float32 round trips and explicit logical byte accounting.

This module is a numerical reference. Decoded copies are kept on CPU; it is
not an implementation of the proposed resident GPU cache.
"""
from __future__ import annotations

import ctypes as ct
from dataclasses import dataclass, field
from pathlib import Path
import math
import struct
import time

import numpy as np


def align(n: int, a: int) -> int:
    return (n + a - 1) // a * a


class AstcEncoder:
    def __init__(self, block: int, quality: float = 98, hdr: bool = False):
        path = Path(__file__).with_name("libakv_astc.dylib" if __import__("sys").platform == "darwin" else "libakv_astc.so")
        self.lib = ct.CDLL(str(path))
        lib = self.lib
        lib.akv_create.argtypes = [ct.c_uint, ct.c_uint, ct.c_float, ct.c_int, ct.POINTER(ct.c_int)]
        lib.akv_create.restype = ct.c_void_p
        lib.akv_destroy.argtypes = [ct.c_void_p]
        lib.akv_error.argtypes = [ct.c_int]
        lib.akv_error.restype = ct.c_char_p
        lib.akv_encode.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_uint, ct.c_uint, ct.c_void_p, ct.c_size_t]
        lib.akv_decode.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_size_t, ct.c_void_p, ct.c_uint, ct.c_uint]
        error = ct.c_int()
        self.ctx = lib.akv_create(block, block, quality, int(hdr), ct.byref(error))
        self._check(error.value)
        if not self.ctx:
            raise RuntimeError("astcenc context allocation failed")
        self.block = block

    def _check(self, code: int):
        if code:
            raise RuntimeError(self.lib.akv_error(code).decode())

    def roundtrip(self, image: np.ndarray) -> tuple[bytes, np.ndarray, float]:
        image = np.ascontiguousarray(image, dtype=np.float32)
        h, w, components = image.shape
        if components != 4 or not np.all(np.isfinite(image)):
            raise ValueError("Expected finite HxWx4 input")
        n = math.ceil(w/self.block) * math.ceil(h/self.block) * 16
        payload = np.empty(n, np.uint8)
        decoded = np.empty_like(image)
        t0 = time.perf_counter()
        self._check(self.lib.akv_encode(self.ctx,image.ctypes.data,w,h,payload.ctypes.data,n))
        encode_s = time.perf_counter()-t0
        self._check(self.lib.akv_decode(self.ctx,payload.ctypes.data,n,decoded.ctypes.data,w,h))
        if not np.all(np.isfinite(decoded)):
            raise ValueError("Non-finite ASTC reconstruction")
        return payload.tobytes(), decoded, encode_s

    def close(self):
        if getattr(self, "ctx", None):
            self.lib.akv_destroy(self.ctx)
            self.ctx = None

    def __del__(self):
        self.close()


def fp16_floor(x):
    x = np.asarray(x, np.float32)
    h = x.astype(np.float16)
    return np.where(h.astype(np.float32) > x,
                    np.nextafter(h, np.float16(-np.inf)), h).astype(np.float16)


def fp16_ceil(x):
    x = np.asarray(x, np.float32)
    h = x.astype(np.float16)
    return np.where(h.astype(np.float32) < x,
                    np.nextafter(h, np.float16(np.inf)), h).astype(np.float16)


@dataclass
class EncodedTensor:
    decoded: np.ndarray
    payloads: list[bytes]
    width: int
    height: int
    block: int
    layout: str
    profile: str
    normalization: str
    offsets: np.ndarray
    scales: np.ndarray | None
    page_tokens: int
    byte_counts: dict[str, int]
    encode_cpu_s: float
    permutation: str | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def logical_bytes(self):
        return sum(self.byte_counts.values())

    @property
    def bits_per_scalar(self):
        return self.logical_bytes * 8 / self.decoded.size

    def astc_file(self, head: int) -> bytes:
        def u24(v):
            return bytes((v & 255, (v >> 8) & 255, (v >> 16) & 255))
        return struct.pack("<I", 0x5CA1AB13) + bytes((self.block,self.block,1)) + u24(self.width) + u24(self.height) + u24(1) + self.payloads[head]


def encode_tensor(x: np.ndarray, config: dict, page_tokens: int = 96,
                  quality: float = 98, seed: int = 20261028) -> EncodedTensor:
    """Encode [H,T,D]. Call only on already completed historical pages."""
    x = np.asarray(x, np.float32)
    if x.ndim != 3 or not np.all(np.isfinite(x)):
        raise ValueError("Expected finite [H,T,D]")
    heads, tokens, dim = x.shape
    block = int(config["block"])
    layout = config["layout"]
    profile = config.get("profile", "ldr")
    normalization = config["normalization"]
    permutation = config.get("permutation")
    if page_tokens % block:
        raise ValueError("Page boundary must align to ASTC block rows")
    if layout not in ("scalar", "rgba") or (layout == "rgba" and dim % 4):
        raise ValueError("Unsupported scalar/component layout")
    if tokens % page_tokens:
        raise ValueError("Only completed pages enter encoder; keep incomplete page in tail")
    if profile == "hdr" and (layout != "scalar" or normalization != "block_min"):
        raise ValueError("HDR pilot supports scalar block-min only")
    comp = 1 if layout == "scalar" else 4
    width, height = align(dim//comp, block), tokens
    work = x.copy()
    rng = np.random.default_rng(seed)
    p_tokens = np.arange(tokens)
    p_channels = np.arange(dim)
    map_bytes = 0
    if permutation == "tokens_within_page":
        perm = rng.permutation(page_tokens)
        p_tokens = (np.arange(tokens//page_tokens)[:,None]*page_tokens+perm[None]).ravel()
        work = work[:,p_tokens,:]
        map_bytes = page_tokens * 2  # fixed map shared by heads/pages
    elif permutation == "channels":
        p_channels = rng.permutation(dim)
        work = work[:,:,p_channels]
        map_bytes = dim * 2
    elif permutation is not None:
        raise ValueError("Unknown permutation")
    pages = tokens//page_tokens
    if normalization == "page_channel":
        view = work.reshape(heads,pages,page_tokens,dim)
        lo = view.min(axis=2)
        hi = view.max(axis=2)
        offset = fp16_floor(lo)
        scale = fp16_ceil(np.maximum(hi-offset.astype(np.float32),np.float32(2**-24)))
        normalized = ((view-offset[:,:,None,:].astype(np.float32))/scale[:,:,None,:].astype(np.float32)).reshape(work.shape)
    elif normalization == "token":
        lo, hi = work.min(-1,keepdims=True), work.max(-1,keepdims=True)
        offset = fp16_floor(lo)
        scale = fp16_ceil(np.maximum(hi-offset.astype(np.float32),np.float32(2**-24)))
        normalized = (work-offset.astype(np.float32))/scale.astype(np.float32)
    elif normalization == "block_min":
        padded = np.pad(work,((0,0),(0,0),(0,width-dim)),mode="edge")
        view = padded.reshape(heads,height//block,block,width//block,block)
        offset = fp16_floor(view.min(axis=(2,4)))
        scale = None
        normalized = (view-offset[:,:,None,:,None].astype(np.float32)).reshape(heads,height,width)
    else:
        raise ValueError("Unknown normalization")
    if not np.all(np.isfinite(offset)) or (scale is not None and not np.all(np.isfinite(scale))):
        raise ValueError("FP16 metadata overflow")
    if profile == "ldr":
        if normalized.min() < -1e-5 or normalized.max() > 1+1e-5:
            raise AssertionError("FP16 rounded metadata fails range coverage")
        normalized = np.clip(normalized,0,1)
    if layout == "scalar":
        scalar = np.pad(normalized,((0,0),(0,0),(0,width-normalized.shape[-1])),mode="edge")
        images = np.ones((heads,height,width,4),np.float32)
        images[...,:3] = scalar[...,None]
    else:
        rgba = normalized.reshape(heads,tokens,dim//4,4)
        images = np.pad(rgba,((0,0),(0,0),(0,width-dim//4),(0,0)),mode="edge")
    encoder = AstcEncoder(block, quality, profile=="hdr")
    payloads, reconstructed, elapsed = [], [], 0.0
    try:
        for im in images:
            payload, dec, seconds = encoder.roundtrip(im)
            payloads.append(payload)
            reconstructed.append(dec)
            elapsed += seconds
    finally:
        encoder.close()
    rec = np.stack(reconstructed)
    rec = rec[...,0] if layout=="scalar" else rec[:,:,:dim//4,:].reshape(heads,tokens,dim)
    if normalization=="block_min":
        rec = (rec.reshape(heads,height//block,block,width//block,block)+offset[:,:,None,:,None].astype(np.float32)).reshape(heads,height,width)[:,:,:dim]
    elif normalization=="page_channel":
        rec = rec[:,:,:dim].reshape(heads,pages,page_tokens,dim)*scale[:,:,None,:].astype(np.float32)+offset[:,:,None,:].astype(np.float32)
        rec = rec.reshape(heads,tokens,dim)
    else:
        rec = rec[:,:,:dim]*scale.astype(np.float32)+offset.astype(np.float32)
    # Inverse permutations restore the original token/channel semantics.
    rec = rec[:,np.argsort(p_tokens),:][:,:,np.argsort(p_channels)]
    counts = {"astc_blocks_including_edge_padding":sum(map(len,payloads)),
              "fp16_offsets":offset.nbytes,"fp16_scales":0 if scale is None else scale.nbytes,
              "permutation_map":map_bytes}
    return EncodedTensor(np.ascontiguousarray(rec),payloads,width,height,block,layout,profile,
                         normalization,offset,scale,page_tokens,counts,elapsed,permutation,
                         {"quality":quality,"original_scalars":x.size,"file_header_bytes":16*heads,
                          "logical_bytes_scope":"blocks+metadata+fixed_permutation; no allocation/staging/tail"})


def affine_quantize(x: np.ndarray, bits: int, axis: str = "token", group: int = 64):
    """Independent numerical affine baseline, not a KIVI/MLX reproduction.

    Codes are actually bit-packed for byte accounting. FP16 scale and offset
    are rounded before code assignment. Returned arrays are CPU reconstructions.
    """
    x = np.asarray(x,np.float32)
    h,t,d = x.shape
    if bits not in (2,4,8):
        raise ValueError("Supported bit widths: 2,4,8")
    if axis=="token":
        if d%group: raise ValueError("group must divide channels")
        view = x.reshape(h,t,d//group,group)
        group_axis = -1
    elif axis=="channel":
        if t%group: raise ValueError("group must divide tokens")
        view = x.reshape(h,t//group,group,d)
        group_axis = 2
    else:
        raise ValueError("Unknown quantization axis")
    levels = (1<<bits)-1
    offset = fp16_floor(view.min(axis=group_axis,keepdims=True))
    maximum = view.max(axis=group_axis,keepdims=True)
    scale = fp16_ceil(np.maximum((maximum-offset.astype(np.float32))/levels,np.float32(2**-24)))
    code = np.clip(np.rint((view-offset.astype(np.float32))/scale.astype(np.float32)),0,levels).astype(np.uint8)
    flat = code.ravel()
    perbyte = 8//bits
    if flat.size%perbyte: flat = np.pad(flat,(0,perbyte-flat.size%perbyte))
    packed = np.zeros(flat.size//perbyte,np.uint8)
    for i in range(perbyte): packed |= flat[i::perbyte] << (i*bits)
    dec = (code.astype(np.float32)*scale.astype(np.float32)+offset.astype(np.float32)).reshape(x.shape)
    counts = {"packed_codes":packed.nbytes,"fp16_offsets":offset.nbytes,"fp16_scales":scale.nbytes}
    return dec,counts


def attention(q,k,v,positions):
    """Return GQA attention output and probabilities for arbitrary query positions."""
    hq,n,d = q.shape
    hk,t,kd = k.shape
    if v.shape != k.shape or kd != d or hq % hk:
        raise ValueError("Incompatible attention shapes")
    qq = q.reshape(hk,hq//hk,n,d)
    scores = np.matmul(qq,k[:,None].swapaxes(-1,-2))*np.float32(d**-0.5)
    mask = np.arange(t)[None,:] <= np.asarray(positions)[:,None]
    scores = np.where(mask[None,None],scores,np.float32(-np.inf))
    scores-=scores.max(-1,keepdims=True)
    p = np.exp(scores)
    p/=p.sum(-1,keepdims=True)
    out = np.matmul(p,v[:,None]).reshape(hq,n,d)
    return out,p.reshape(hq,n,t)


def errors(reference, estimate):
    a,b = np.asarray(reference,np.float64),np.asarray(estimate,np.float64)
    err = b-a
    return {"mse":float(np.mean(err*err)),"relative_l2":float(np.linalg.norm(err.ravel())/max(np.linalg.norm(a.ravel()),1e-30)),
            "max_abs":float(np.max(np.abs(err))),"cosine":float(np.sum(a*b)/max(np.linalg.norm(a.ravel())*np.linalg.norm(b.ravel()),1e-30))}


def attention_metrics(reference, estimate, reference_probs, estimate_probs):
    metrics = errors(reference,estimate)
    a,b = reference_probs.astype(np.float64),estimate_probs.astype(np.float64)
    kl = np.sum(a*(np.log(np.maximum(a,1e-300))-np.log(np.maximum(b,1e-300))),axis=-1)
    metrics.update(mean_kl_nats=float(np.mean(kl)),p95_kl_nats=float(np.percentile(kl,95)),
                   mean_total_variation=float(np.mean(np.abs(a-b).sum(-1)*0.5)))
    return metrics
