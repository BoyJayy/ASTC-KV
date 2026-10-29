#include <metal_stdlib>
using namespace metal;

// Experimental eight-token SIMD kernel; D64 only. One SIMD group processes
// one (query head, context split). A second dispatch merges exact softmax states.
// Both the ASTC and ordinary-texture variants use integer texel coordinates.
// 0: normalized texture, 1: restored FP16 buffer, 2: normalized FP16 buffer.
constant uint storage_path [[function_constant(0)]];
constant bool factor_v [[function_constant(1)]];
// Host supplies false for ordinary paths and true for compact atlas paths.
constant bool compact_atlas [[function_constant(2)]];
// 0: unchanged v6 addressing, 1: hoisted channel maps, 2: recurrent token row.
constant uint address_mode [[function_constant(3)]];

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

// Each of four channels is mapped independently: float4 loads may cross a
// 6-channel ASTC block boundary or the 32-block-wide compact row boundary.
inline float4 compactTexture4(texture2d_array<float, access::read> tex,
                              uint head,uint token,uint channel,constant Params &p) {
    float4 result;
    for (uint j=0;j<4;++j) {
        uint c=channel+j;
        uint block=(head*(p.height/6)+token/6)*(p.width/6)+c/6;
        uint2 xy=uint2((block%32)*6+c%6,(block/32)*6+token%6);
        result[j]=tex.read(xy,0,0).r;
    }
    return result;
}

// Channel maps are computed once per lane before the token loop. Keep every
// scalar address independent across 6-wide block and 32-block atlas seams.
inline float4 compactTexture4Mapped(texture2d_array<float, access::read> tex,
                                    uint tokenBlockBase,uint tokenWithinY,
                                    uint4 channelBlock,uint4 channelWithin) {
    uint4 block=uint4(tokenBlockBase)+channelBlock;
    uint4 x=(block & uint4(31u))*6u+channelWithin;
    uint4 y=(block >> uint4(5u))*6u+uint4(tokenWithinY);
    return float4(tex.read(uint2(x.x,y.x),0,0).r,
                  tex.read(uint2(x.y,y.y),0,0).r,
                  tex.read(uint2(x.z,y.z),0,0).r,
                  tex.read(uint2(x.w,y.w),0,0).r);
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
    uint4 channelBlock[4],channelWithin[4];
    uint compactHeadBase=0;
    uint recurrentBlockBase=0,recurrentWithinY=0;
    if (address_mode > 0) {
        #pragma unroll
        for (uint j=0;j<4;++j) {
            uint c=quarter*16+j*4;
            uint4 channels=uint4(c,c+1,c+2,c+3);
            channelBlock[j]=channels/6u;
            channelWithin[j]=channels-channelBlock[j]*6u;
        }
        compactHeadBase=kh*(p.height/6u)*11u;
        if (address_mode == 2) {
            uint initialToken=start+token_lane;
            recurrentBlockBase=compactHeadBase+(initialToken/6u)*11u;
            recurrentWithinY=initialToken%6u;
        }
    }
    for (uint base = start; base < end; base += 8) {
        uint t = base + token_lane;
        bool valid = t < end;
        uint tokenBlockBase=0,tokenWithinY=0;
        if (address_mode == 1) {
            tokenBlockBase=compactHeadBase+(t/6u)*11u;
            tokenWithinY=t%6u;
        } else if (address_mode == 2) {
            tokenBlockBase=recurrentBlockBase;
            tokenWithinY=recurrentWithinY;
        }
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
                    float4 u = storage_path == 0 ? (compact_atlas ? (address_mode == 0 ? compactTexture4(value_texture,kh,t,c,p) : compactTexture4Mapped(value_texture,tokenBlockBase,tokenWithinY,channelBlock[j],channelWithin[j])) : texture4(value_texture,kh,t,c,p)) : load4h(value_buffer+i);
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
        if (address_mode == 2) {
            // t advances by 8: one block row plus a second carry iff rem >= 4.
            // All lanes advance, including inactive lanes in the final wave.
            bool carry=recurrentWithinY>=4u;
            recurrentBlockBase+=11u*(1u+uint(carry));
            recurrentWithinY=carry?recurrentWithinY-4u:recurrentWithinY+2u;
        }
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

// Correctness-only independent scalar address calculation; original decoder
// remains unchanged. p.width/p.height describe the source, not compact atlas.
kernel void decode_compact_texture(
    constant Params &p [[buffer(0)]],
    device float *output [[buffer(1)]],
    texture2d_array<float, access::read> tex [[texture(0)]],
    uint i [[thread_position_in_grid]]) {
    uint count=p.kv_heads*p.tokens*p.dim;
    if(i>=count) return;
    uint channel=i%p.dim;
    uint token=(i/p.dim)%p.tokens;
    uint head=i/(p.dim*p.tokens);
    uint blocks_per_source_row=p.width/6;
    uint blocks_per_head=blocks_per_source_row*(p.height/6);
    uint original_block=head*blocks_per_head+(token/6)*blocks_per_source_row+channel/6;
    uint atlas_row=original_block/32;
    uint atlas_column=original_block-atlas_row*32;
    uint2 coordinate=uint2(atlas_column*6+channel%6,atlas_row*6+token%6);
    output[i]=tex.read(coordinate,0,0).r;
}


// V8 changes only paired V loading. Original ten paths above remain unchanged.
constant bool paired_use_gather [[function_constant(4)]];

inline uint2 compactPairXY(uint head,uint token,uint channel,constant Params &p) {
    uint block=(head*(p.height/6u)+token/6u)*11u+channel/6u;
    return uint2((block&31u)*6u+channel%6u,(block>>5u)*6u+token%6u);
}

inline float4 compactRead4Sample(texture2d_array<float,access::sample> tex,
                                uint head,uint token,uint channel,constant Params &p) {
    float4 result;
    for(uint j=0;j<4;++j) result[j]=tex.read(compactPairXY(head,token,channel+j,p),0u,0u).r;
    return result;
}

// MSL gather ordering at the footprint center: (-,+), (+,+), (+,-), (-,-).
// In integer texture coordinates these are odd-left, odd-right, even-right, even-left.
inline float4 compactReadQuad(texture2d_array<float,access::sample> tex,uint2 xy) {
    return float4(tex.read(xy+uint2(0,1),0u,0u).r,
                  tex.read(xy+uint2(1,1),0u,0u).r,
                  tex.read(xy+uint2(1,0),0u,0u).r,
                  tex.read(xy,0u,0u).r);
}

inline float4 compactGatherQuad(texture2d_array<float,access::sample> tex,
                               uint2 xy,float2 inverseSize) {
    constexpr sampler quadSampler(coord::normalized,address::clamp_to_edge,filter::linear);
    // The footprint center is between four texel centers. This is a gather,
    // not filtered interpolation; no weighted values are produced by sampler.
    float2 position=(float2(xy)+float2(1.0f))*inverseSize;
    return tex.gather(quadSampler,position,0u,int2(0),component::x);
}

kernel void attention_split_paired(
    constant Params &p [[buffer(0)]],
    device const float *query [[buffer(1)]],
    device const half *key_buffer [[buffer(2)]],
    device const half *value_offset [[buffer(6)]],
    device const half *value_scale [[buffer(7)]],
    device float *partial [[buffer(8)]],
    device float2 *statistics [[buffer(9)]],
    texture2d_array<float,access::sample> value_texture [[texture(1)]],
    uint lane [[thread_index_in_simdgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
    uint qh=group.x,split=group.y,quarter=lane%4u,token_lane=lane/4u;
    uint kh=qh/(p.query_heads/p.kv_heads);
    float4 q[4],acc[4];
    for(uint j=0;j<4;++j) {
        q[j]=load4(query+qh*p.dim+quarter*16u+j*4u);
        acc[j]=0.0f;
    }
    float offset_acc=0.0f,m=-INFINITY,denominator=0.0f;
    uint start=split*p.tile_tokens,end=min(start+p.tile_tokens,p.tokens);
    float qk_scale=rsqrt(float(p.dim));
    float2 inverseSize=1.0f/float2(value_texture.get_width(),value_texture.get_height());
    for(uint base=start;base<end;base+=8u) {
        uint t=base+token_lane;
        bool valid=t<end;
        float local_score=0.0f;
        for(uint j=0;j<4;++j) {
            uint c=quarter*16u+j*4u;
            float4 k=valid?load4h(key_buffer+(kh*p.tokens+t)*p.dim+c):float4(0.0f);
            local_score+=dot(q[j],k);
        }
        local_score+=simd_shuffle_xor(local_score,1);
        local_score+=simd_shuffle_xor(local_score,2);
        float score=valid?local_score*qk_scale:-INFINITY;
        float next_m=max(m,simd_max(score));
        float old_weight=exp(m-next_m),new_weight=exp(score-next_m);
        float scale=0.0f,offset=0.0f;
        if(valid) { uint i=kh*p.tokens+t; scale=float(value_scale[i]);offset=float(value_offset[i]); }
        // Every lane participates in shuffles, including invalid tail lanes.
        float peer_weight=simd_shuffle_xor(new_weight,4);
        float peer_scale=simd_shuffle_xor(scale,4);
        float peer_offset=simd_shuffle_xor(offset,4);
        bool owner=valid && (token_lane%2u==0u);
        bool has_peer=t+1u<end;
        // A 2x2 footprint stays in one ASTC6 block only with even global t/c.
        // Odd split starts and missing peers use independent mapped reads.
        bool quad_safe=(t%2u==0u) && has_peer;
        float a=new_weight*scale,b=peer_weight*peer_scale;
        for(uint j=0;j<4;++j) {
            float4 weighted=0.0f;
            if(owner) {
                uint c=quarter*16u+j*4u;
                if(quad_safe) {
                    uint2 xy0=compactPairXY(kh,t,c,p),xy1=compactPairXY(kh,t,c+2u,p);
                    float4 g0=paired_use_gather?compactGatherQuad(value_texture,xy0,inverseSize):compactReadQuad(value_texture,xy0);
                    float4 g1=paired_use_gather?compactGatherQuad(value_texture,xy1,inverseSize):compactReadQuad(value_texture,xy1);
                    weighted=float4(a*g0.w+b*g0.x,a*g0.z+b*g0.y,
                                    a*g1.w+b*g1.x,a*g1.z+b*g1.y);
                } else {
                    float4 u0=compactRead4Sample(value_texture,kh,t,c,p);
                    float4 u1=has_peer?compactRead4Sample(value_texture,kh,t+1u,c,p):float4(0.0f);
                    weighted=u0*a+u1*b;
                }
            }
            // Odd token quads contribute zero. The reduction still sums all 8 quads.
            weighted+=simd_shuffle_xor(weighted,4);
            weighted+=simd_shuffle_xor(weighted,8);
            weighted+=simd_shuffle_xor(weighted,16);
            acc[j]=fma(acc[j],old_weight,weighted);
        }
        float off=owner?(new_weight*offset+peer_weight*peer_offset):0.0f;
        off+=simd_shuffle_xor(off,4);
        off+=simd_shuffle_xor(off,8);
        off+=simd_shuffle_xor(off,16);
        offset_acc=fma(offset_acc,old_weight,off);
        // All original token weights, not pair weights: no doubled denominator.
        float wave_weight=simd_sum(quarter==0?new_weight:0.0f);
        denominator=fma(denominator,old_weight,wave_weight);
        m=next_m;
    }
    uint state=qh*p.splits+split;
    if(token_lane==0) for(uint j=0;j<4;++j)
        store4(partial+state*p.dim+quarter*16u+j*4u,acc[j]+offset_acc);
    if(lane==0) statistics[state]=float2(m,denominator);
}

// Every head, even token anchor and even channel anchor is checked. The last
// odd logical token uses its physically valid padded neighbor for this check.
// Output is [pair][gather xyzw][four direct reads in the documented order].
kernel void check_compact_gather(
    constant Params &p [[buffer(0)]],
    device float *output [[buffer(1)]],
    texture2d_array<float,access::sample> tex [[texture(0)]],
    uint i [[thread_position_in_grid]]) {
    uint channel_pairs=p.dim/2u,token_pairs=(p.tokens+1u)/2u;
    uint count=p.kv_heads*token_pairs*channel_pairs;
    if(i>=count) return;
    uint c=(i%channel_pairs)*2u,t=((i/channel_pairs)%token_pairs)*2u;
    uint h=i/(channel_pairs*token_pairs);
    uint2 xy=compactPairXY(h,t,c,p);
    float2 inverseSize=1.0f/float2(tex.get_width(),tex.get_height());
    float4 gathered=compactGatherQuad(tex,xy,inverseSize);
    float4 direct=compactReadQuad(tex,xy);
    store4(output+i*8u,gathered);
    store4(output+i*8u+4u,direct);
}


// V9 keeps the v7 per-token weight/accumulation schedule. Gather only supplies
// both tokens' normalized V, then an unconditional shuffle returns odd-token V.
constant bool exchange_use_gather [[function_constant(5)]];

inline float4 compactRead4MappedSample(texture2d_array<float,access::sample> tex,
                                       uint tokenBlockBase,uint tokenWithinY,
                                       uint4 channelBlock,uint4 channelWithin) {
    uint4 block=uint4(tokenBlockBase)+channelBlock;
    uint4 x=(block&uint4(31u))*6u+channelWithin;
    uint4 y=(block>>uint4(5u))*6u+uint4(tokenWithinY);
    return float4(tex.read(uint2(x.x,y.x),0u,0u).r,
                  tex.read(uint2(x.y,y.y),0u,0u).r,
                  tex.read(uint2(x.z,y.z),0u,0u).r,
                  tex.read(uint2(x.w,y.w),0u,0u).r);
}

kernel void attention_split_exchange(
    constant Params &p [[buffer(0)]],
    device const float *query [[buffer(1)]],
    device const half *key_buffer [[buffer(2)]],
    device const half *value_offset [[buffer(6)]],
    device const half *value_scale [[buffer(7)]],
    device float *partial [[buffer(8)]],
    device float2 *statistics [[buffer(9)]],
    texture2d_array<float,access::sample> value_texture [[texture(1)]],
    uint lane [[thread_index_in_simdgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
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
    bool factored = true;
    float m = -INFINITY;
    float denominator = 0.0f;
    uint start = split * p.tile_tokens;
    uint end = min(start + p.tile_tokens, p.tokens);
    float qk_scale = rsqrt(float(p.dim));
    uint4 channelBlock[4],channelWithin[4];
    #pragma unroll
    for(uint j=0;j<4;++j) {
        uint c=quarter*16u+j*4u;
        uint4 channels=uint4(c,c+1u,c+2u,c+3u);
        channelBlock[j]=channels/6u;
        channelWithin[j]=channels-channelBlock[j]*6u;
    }
    uint compactHeadBase=kh*(p.height/6u)*11u;
    float2 inverseSize=1.0f/float2(value_texture.get_width(),value_texture.get_height());
    for (uint base = start; base < end; base += 8) {
        uint t = base + token_lane;
        bool valid = t < end;
        uint tokenBlockBase=compactHeadBase+(t/6u)*11u;
        uint tokenWithinY=t%6u;
        uint pairToken=t-(token_lane&1u);
        bool pairSafe=(pairToken%2u==0u) && pairToken+1u<end;
        bool evenOwner=(token_lane&1u)==0u;
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
            float4 ownValues=0.0f,oddValues=0.0f;
            v[j] = 0.0f;
            if(valid) {
                uint i=(kh*p.tokens+t)*p.dim+c;
                k=load4h(key_buffer+i);
            }
            if(pairSafe && evenOwner) {
                uint4 block=uint4(tokenBlockBase)+channelBlock[j];
                uint4 x=(block&uint4(31u))*6u+channelWithin[j];
                uint4 y=(block>>uint4(5u))*6u+uint4(tokenWithinY);
                uint2 xy0=uint2(x.x,y.x),xy1=uint2(x.z,y.z);
                float4 g0=exchange_use_gather?compactGatherQuad(value_texture,xy0,inverseSize):compactReadQuad(value_texture,xy0);
                float4 g1=exchange_use_gather?compactGatherQuad(value_texture,xy1,inverseSize):compactReadQuad(value_texture,xy1);
                ownValues=float4(g0.w,g0.z,g1.w,g1.z);
                oddValues=float4(g0.x,g0.y,g1.x,g1.y);
            }
            // Unconditional across all 32 lanes, including invalid final lanes.
            float4 exchanged=simd_shuffle_xor(oddValues,4);
            if(valid) {
                if(pairSafe) v[j]=evenOwner?ownValues:exchanged;
                else v[j]=compactRead4MappedSample(value_texture,tokenBlockBase,tokenWithinY,channelBlock[j],channelWithin[j]);
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
