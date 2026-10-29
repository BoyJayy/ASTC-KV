// Thin ABI over the unmodified Arm astcenc library. Apache-2.0 upstream.
#include "astcenc.h"
#include <cstddef>
#include <cstdint>

extern "C" {
void* akv_create(unsigned bx, unsigned by, float quality, int hdr, int* error) {
    astcenc_config cfg;
    auto e = astcenc_config_init(hdr ? ASTCENC_PRF_HDR_RGB_LDR_A : ASTCENC_PRF_LDR,
                                bx, by, 1, quality, 0, &cfg);
    astcenc_context* ctx = nullptr;
    if (e == ASTCENC_SUCCESS) e = astcenc_context_alloc(&cfg, 1, &ctx, nullptr);
    *error = static_cast<int>(e);
    return ctx;
}
void akv_destroy(void* ctx) { astcenc_context_free((astcenc_context*)ctx); }
const char* akv_error(int e) { return astcenc_get_error_string((astcenc_error)e); }
int akv_encode(void* ctx, float* pixels, unsigned w, unsigned h,
               uint8_t* compressed, size_t bytes) {
    void* slices[] = {pixels};
    astcenc_image im{w, h, 1, ASTCENC_TYPE_F32, slices};
    astcenc_swizzle sw{ASTCENC_SWZ_R,ASTCENC_SWZ_G,ASTCENC_SWZ_B,ASTCENC_SWZ_A};
    auto e = astcenc_compress_image((astcenc_context*)ctx,&im,&sw,compressed,bytes,0);
    astcenc_compress_reset((astcenc_context*)ctx);
    return static_cast<int>(e);
}
int akv_decode(void* ctx, const uint8_t* compressed, size_t bytes,
               float* pixels, unsigned w, unsigned h) {
    void* slices[] = {pixels};
    astcenc_image im{w,h,1,ASTCENC_TYPE_F32,slices};
    astcenc_swizzle sw{ASTCENC_SWZ_R,ASTCENC_SWZ_G,ASTCENC_SWZ_B,ASTCENC_SWZ_A};
    auto e = astcenc_decompress_image((astcenc_context*)ctx,compressed,bytes,&im,&sw,0);
    astcenc_decompress_reset((astcenc_context*)ctx);
    return static_cast<int>(e);
}
}
