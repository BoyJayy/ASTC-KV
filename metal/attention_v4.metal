#include <metal_stdlib>
using namespace metal;

// Experimental eight-token SIMD kernel; D64 only. One SIMD group processes
// one (query head, context split). A second dispatch merges exact softmax states.
// Both the ASTC and ordinary-texture variants use integer texel coordinates.
// 0: normalized texture, 1: restored FP16 buffer, 2: normalized FP16 buffer.
constant uint storage_path [[function_constant(0)]];
constant bool factor_v [[function_constant(1)]];

struct Params {
    uint tokens;
    uint query_heads;
    uint kv_heads;
    uint dim;
    uint page_tokens;
    uint pages;
    uint tile_tokens;
    uint splits;
    uint components;  // 1: scalar .r, 4: packed .rgba
    uint width;
    uint height;
    uint k_storage;        // 0: ASTC/normalized texture, 1: FP16 buffer
    uint v_storage;
    uint k_normalization;  // 0: per-page/channel, 1: per-token scalar
    uint v_normalization;
};

inline float4 load4(device const float *p) {
    return float4(p[0], p[1], p[2], p[3]);
}

inline float4 load4h(device const half *p) {
    return float4(p[0], p[1], p[2], p[3]);
}

inline void store4(device float *p, float4 v) {
    p[0] = v.x; p[1] = v.y; p[2] = v.z; p[3] = v.w;
}

inline float4 texture4(texture2d_array<float, access::read> tex,
                      uint head, uint token, uint channel,
                      constant Params &p) {
    if (p.components == 4) {
        return tex.read(uint2(channel / 4, token), head, 0);
    }
    return float4(tex.read(uint2(channel + 0, token), head, 0).r,
                  tex.read(uint2(channel + 1, token), head, 0).r,
                  tex.read(uint2(channel + 2, token), head, 0).r,
                  tex.read(uint2(channel + 3, token), head, 0).r);
}

inline float4 affine4(float4 normalized,
                       device const half *offset,
                       device const half *scale,
                       uint head, uint token, uint channel,
                       uint normalization,
                       constant Params &p) {
    if (normalization == 1) {
        uint i = head * p.tokens + token;
        return fma(normalized,
                   float4(scale[i]), float4(offset[i]));
    }
    uint i = (head * p.pages + token / p.page_tokens) * p.dim + channel;
    return fma(normalized,
               load4h(scale + i), load4h(offset + i));
}

kernel void attention_split(
    constant Params &p [[buffer(0)]],
    device const float *query [[buffer(1)]],
    device const half *key_buffer [[buffer(2)]],
    device const half *value_buffer [[buffer(3)]],
    device const half *key_offset [[buffer(4)]],
    device const half *key_scale [[buffer(5)]],
    device const half *value_offset [[buffer(6)]],
    device const half *value_scale [[buffer(7)]],
    device float *partial [[buffer(8)]],
    device float2 *statistics [[buffer(9)]],
    texture2d_array<float, access::read> key_texture [[texture(0)]],
    texture2d_array<float, access::read> value_texture [[texture(1)]],
    uint lane [[thread_index_in_simdgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
    // D=64 specialization: four lanes per token and eight tokens per SIMD wave.
    // Each lane holds sixteen channels. Quad reduction forms eight scores;
    // cross-quad reduction forms four sixteen-channel value accumulators.
    uint qh = group.x;
    uint split = group.y;
    uint quarter = lane % 4;
    uint token_lane = lane / 4;
    uint kh = qh / (p.query_heads / p.kv_heads);
    float4 q[4];
    float4 acc[4];
    for (uint j=0;j<4;++j) {
        q[j] = load4(query + qh * p.dim + quarter * 16 + j * 4);
        acc[j] = 0.0f;
    }
    float offset_acc = 0.0f;
    bool factored = factor_v && storage_path != 1 && p.v_storage == 0 && p.v_normalization == 1;
    float m = -INFINITY;
    float denominator = 0.0f;
    uint start = split * p.tile_tokens;
    uint end = min(start + p.tile_tokens, p.tokens);
    float qk_scale = rsqrt(float(p.dim));
    for (uint base = start; base < end; base += 8) {
        uint t = base + token_lane;
        bool valid = t < end;
        float local_score = 0.0f;
        float4 v[4];
        float vs = 1.0f, vo = 0.0f;
        if (valid && factored) {
            uint mi = kh * p.tokens + t;
            vs = float(value_scale[mi]);
            vo = float(value_offset[mi]);
        }
        for (uint j=0;j<4;++j) {
            uint c = quarter * 16 + j * 4;
            float4 k = 0.0f;
            v[j] = 0.0f;
            if (valid) {
                uint i = (kh * p.tokens + t) * p.dim + c;
                if (storage_path != 1 && p.k_storage == 0) {
                    float4 u = storage_path == 0 ? texture4(key_texture,kh,t,c,p) : load4h(key_buffer+i);
                    k = affine4(u,key_offset,key_scale,kh,t,c,p.k_normalization,p);
                } else k = load4h(key_buffer+i);
                if (storage_path != 1 && p.v_storage == 0) {
                    float4 u = storage_path == 0 ? texture4(value_texture,kh,t,c,p) : load4h(value_buffer+i);
                    v[j] = factored ? u : affine4(u,value_offset,value_scale,kh,t,c,p.v_normalization,p);
                } else v[j] = load4h(value_buffer+i);
            }
            local_score += dot(q[j],k);
        }
        local_score += simd_shuffle_xor(local_score,1);
        local_score += simd_shuffle_xor(local_score,2);
        float score = valid ? local_score * qk_scale : -INFINITY;
        float next_m = max(m,simd_max(score));
        float old_weight = exp(m-next_m);
        float new_weight = exp(score-next_m);
        for (uint j=0;j<4;++j) {
            float4 weighted = v[j] * (new_weight * vs);
            weighted += simd_shuffle_xor(weighted,4);
            weighted += simd_shuffle_xor(weighted,8);
            weighted += simd_shuffle_xor(weighted,16);
            acc[j] = fma(acc[j],old_weight,weighted);
        }
        if (factored) {
            float off = new_weight * vo;
            off += simd_shuffle_xor(off,4);
            off += simd_shuffle_xor(off,8);
            off += simd_shuffle_xor(off,16);
            offset_acc = fma(offset_acc,old_weight,off);
        }
        float wave_weight = simd_sum(quarter == 0 ? new_weight : 0.0f);
        denominator = fma(denominator,old_weight,wave_weight);
        m = next_m;
    }
    uint state = qh * p.splits + split;
    if (token_lane == 0) {
        for (uint j=0;j<4;++j) store4(partial + state*p.dim + quarter*16+j*4,acc[j]+offset_acc);
    }
    if (lane == 0) statistics[state] = float2(m,denominator);
}

kernel void attention_merge(
    constant Params &p [[buffer(0)]],
    device const float *partial [[buffer(8)]],
    device const float2 *statistics [[buffer(9)]],
    device float *output [[buffer(10)]],
    uint lane [[thread_index_in_simdgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
    uint qh = group.x;
    uint c = lane * 4;
    if (c >= p.dim) return;
    float m = -INFINITY;
    for (uint s = 0; s < p.splits; ++s) {
        m = max(m, statistics[qh * p.splits + s].x);
    }
    float4 acc = 0.0f;
    float denominator = 0.0f;
    for (uint s = 0; s < p.splits; ++s) {
        uint state = qh * p.splits + s;
        float2 stat = statistics[state];
        float weight = exp(stat.x - m);
        denominator = fma(stat.y, weight, denominator);
        acc = fma(load4(partial + state * p.dim + c), weight, acc);
    }
    store4(output + qh * p.dim + c, acc / denominator);
}

// Correctness-only readback, never dispatched in timed attention. This lets the
// CPU reference consume exactly the normalized values returned by the GPU.
kernel void decode_texture(
    constant Params &p [[buffer(0)]],
    device float *output [[buffer(1)]],
    texture2d_array<float, access::read> tex [[texture(0)]],
    uint i [[thread_position_in_grid]]) {
    uint count = p.kv_heads * p.tokens * p.dim;
    if (i >= count) return;
    uint c = i % p.dim;
    uint t = (i / p.dim) % p.tokens;
    uint h = i / (p.dim * p.tokens);
    float4 rgba = tex.read(uint2(c / p.components, t), h, 0);
    output[i] = p.components == 4 ? rgba[c % 4] : rgba.r;
}
