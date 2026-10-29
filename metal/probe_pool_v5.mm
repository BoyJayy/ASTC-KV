#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <CommonCrypto/CommonDigest.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include <sys/resource.h>

// V5 resident-pool instrument; unchanged v4 shader and original seven-path gates.
// A precompressed-cache diagnostic, not a runtime integration or strong baseline.
struct Params {
    uint32_t tokens, query_heads, kv_heads, dim, page_tokens, pages;
    uint32_t tile_tokens, splits, components, width, height;
    uint32_t k_storage, v_storage, k_normalization, v_normalization;
};
static_assert(sizeof(Params) == 60, "Keep host/Metal parameter ABI identical");

using Clock = std::chrono::steady_clock;
static double elapsedMS(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
static void require(bool ok, const std::string &message) {
    if (!ok) throw std::runtime_error(message);
}
static std::string str(NSString *s) { return s ? s.UTF8String : "<nil>"; }
static NSUInteger alignUp(NSUInteger value, NSUInteger alignment) {
    return (value + alignment - 1) / alignment * alignment;
}
static uint32_t u24(const uint8_t *p) {
    return uint32_t(p[0]) | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16);
}
static NSData *fileData(NSString *base, id relative, NSUInteger expected = 0) {
    require([relative isKindOfClass:[NSString class]], "Expected a relative file path");
    NSString *path = [base stringByAppendingPathComponent:(NSString *)relative];
    NSError *error = nil;
    NSData *data = [NSData dataWithContentsOfFile:path options:0 error:&error];
    require(data != nil, "Cannot read " + str(path) + ": " + str(error.localizedDescription));
    if (expected) require(data.length == expected, "Wrong byte length: " + str(path));
    return data;
}
static id<MTLBuffer> bufferWithData(id<MTLDevice> device, NSData *data, NSString *label) {
    id<MTLBuffer> result = [device newBufferWithBytes:data.bytes length:data.length
                                            options:MTLResourceStorageModeShared];
    require(result != nil, "Cannot allocate " + str(label));
    result.label = label;
    return result;
}
static id<MTLBuffer> bufferWithSize(id<MTLDevice> device, NSUInteger size, NSString *label) {
    id<MTLBuffer> result = [device newBufferWithLength:size options:MTLResourceStorageModeShared];
    require(result != nil, "Cannot allocate " + str(label));
    result.label = label;
    return result;
}
static id<MTLComputePipelineState> pipeline(id<MTLDevice> device, id<MTLLibrary> library,
                                           NSString *name, int storagePath = -1, bool factored = false) {
    NSError *error = nil;
    id<MTLFunction> function = nil;
    if (storagePath >= 0) {
        MTLFunctionConstantValues *constants = [MTLFunctionConstantValues new];
        uint32_t value = uint32_t(storagePath);
        [constants setConstantValue:&value type:MTLDataTypeUInt atIndex:0];
        [constants setConstantValue:&factored type:MTLDataTypeBool atIndex:1];
        function = [library newFunctionWithName:name constantValues:constants error:&error];
    } else {
        function = [library newFunctionWithName:name];
    }
    require(function != nil, "Cannot load function " + str(name) + ": " + str(error.localizedDescription));
    id<MTLComputePipelineState> result = [device newComputePipelineStateWithFunction:function error:&error];
    require(result != nil, "Cannot create pipeline " + str(name) + ": " + str(error.localizedDescription));
    require(result.threadExecutionWidth == 32, "Probe requires a 32-lane SIMD group");
    return result;
}
static void complete(id<MTLCommandBuffer> command) {
    [command commit];
    [command waitUntilCompleted];
    require(command.status == MTLCommandBufferStatusCompleted,
            "GPU command failed: " + str(command.error.localizedDescription));
}
static id gpuMS(id<MTLCommandBuffer> command) {
    if (command.GPUEndTime <= command.GPUStartTime || command.GPUStartTime == 0)
        return [NSNull null];
    return @((command.GPUEndTime - command.GPUStartTime) * 1000.0);
}
static MTLPixelFormat astcFormat(uint32_t x, uint32_t y) {
    if (x == 4 && y == 4) return MTLPixelFormatASTC_4x4_LDR;
    if (x == 6 && y == 6) return MTLPixelFormatASTC_6x6_LDR;
    if (x == 8 && y == 8) return MTLPixelFormatASTC_8x8_LDR;
    throw std::runtime_error("Probe supports ASTC LDR 4x4, 6x6, or 8x8 only");
}
struct ASTCHeader {
    uint32_t bx, by, width, height;
    NSUInteger rowBytes, blockRows, payloadBytes;
};
static ASTCHeader astcHeader(NSData *data) {
    require(data.length >= 16, "Truncated .astc header");
    const uint8_t *p = (const uint8_t *)data.bytes;
    require(p[0] == 0x13 && p[1] == 0xab && p[2] == 0xa1 && p[3] == 0x5c,
            "Wrong .astc magic");
    require(p[6] == 1 && u24(p + 13) == 1, "Only 2D ASTC files are supported");
    ASTCHeader h{p[4], p[5], u24(p + 7), u24(p + 10), 0, 0, 0};
    (void)astcFormat(h.bx, h.by);
    require(h.width > 0 && h.height > 0, "Zero texture dimensions");
    h.rowBytes = ((h.width + h.bx - 1) / h.bx) * 16;
    h.blockRows = (h.height + h.by - 1) / h.by;
    h.payloadBytes = h.rowBytes * h.blockRows;
    require(data.length == 16 + h.payloadBytes, "Wrong ASTC payload size");
    return h;
}
struct Uploaded {
    id<MTLTexture> texture;
    id<MTLBuffer> staging;
    NSUInteger payloadBytes, fileBytes, stagingBytes;
    ASTCHeader header;
};
static Uploaded uploadASTC(id<MTLDevice> device, id<MTLBlitCommandEncoder> blit,
                           NSString *base, NSArray *files, NSString *label,
                           const Params &p) {
    require([files isKindOfClass:[NSArray class]] && files.count == p.kv_heads,
            "Each ASTC file array must have kv_heads entries");
    NSMutableArray<NSData *> *images = [NSMutableArray new];
    for (id file in files) [images addObject:fileData(base, file)];
    ASTCHeader h = astcHeader(images[0]);
    require(h.width >= (p.dim / p.components) && h.height >= p.tokens,
            "ASTC atlas is too small for the declared tensor");
    require(h.height == p.pages * p.page_tokens,
            "ASTC height must equal ceil(tokens/page_tokens)*page_tokens");
    require(h.width == alignUp(p.dim / p.components, h.bx),
            "ASTC width must pad D/components to the selected block width");
    require(p.page_tokens % h.by == 0, "Page height must be ASTC block aligned");
    for (NSData *data in images) {
        ASTCHeader hi = astcHeader(data);
        require(hi.width == h.width && hi.height == h.height && hi.bx == h.bx && hi.by == h.by,
                "All head images must have identical dimensions and block footprint");
    }
    MTLTextureDescriptor *desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:astcFormat(h.bx,h.by)
                                                                                width:h.width height:h.height mipmapped:NO];
    desc.textureType = MTLTextureType2DArray;
    desc.arrayLength = p.kv_heads;
    desc.storageMode = MTLStorageModePrivate;
    desc.usage = MTLTextureUsageShaderRead;
    id<MTLTexture> tex = [device newTextureWithDescriptor:desc];
    require(tex != nil, "Device rejected " + str(label) + " ASTC texture allocation");
    tex.label = label;
    // 256 is a conservative staging stride; it is not a compressed payload byte
    // count. Keep this distinction in both output and memory accounting.
    NSUInteger paddedRow = alignUp(h.rowBytes, 256);
    NSUInteger sliceBytes = paddedRow * h.blockRows;
    id<MTLBuffer> staging = bufferWithSize(device, sliceBytes * p.kv_heads,
                                          [label stringByAppendingString:@" staging"]);
    std::memset(staging.contents, 0, staging.length);
    for (uint32_t head = 0; head < p.kv_heads; ++head) {
        const uint8_t *source = (const uint8_t *)images[head].bytes + 16;
        uint8_t *dest = (uint8_t *)staging.contents + head * sliceBytes;
        for (NSUInteger row = 0; row < h.blockRows; ++row)
            std::memcpy(dest + row*paddedRow, source + row*h.rowBytes, h.rowBytes);
        [blit copyFromBuffer:staging sourceOffset:head*sliceBytes
          sourceBytesPerRow:paddedRow sourceBytesPerImage:0
                 sourceSize:MTLSizeMake(h.width,h.height,1) toTexture:tex
           destinationSlice:head destinationLevel:0 destinationOrigin:MTLOriginMake(0,0,0)];
    }
    return {tex, staging, h.payloadBytes*p.kv_heads, (h.payloadBytes+16)*p.kv_heads,
            staging.length, h};
}

static std::vector<float> decode(id<MTLDevice> device, id<MTLCommandQueue> queue,
                                 id<MTLComputePipelineState> pipe,
                                 id<MTLTexture> texture, const Params &p) {
    NSUInteger count = NSUInteger(p.kv_heads) * p.tokens * p.dim;
    id<MTLBuffer> out = bufferWithSize(device,count*sizeof(float),@"diagnostic normalized readback");
    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [command computeCommandEncoder];
    [enc setComputePipelineState:pipe];
    [enc setBytes:&p length:sizeof(p) atIndex:0];
    [enc setBuffer:out offset:0 atIndex:1];
    [enc setTexture:texture atIndex:0];
    [enc dispatchThreads:MTLSizeMake(count,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    [enc endEncoding];
    complete(command);
    const float *values = (const float *)out.contents;
    return std::vector<float>(values, values + count);
}
static std::vector<float> restore(const std::vector<float> &normalized,
                                  id<MTLBuffer> offset, id<MTLBuffer> scale,
                                  uint32_t normalization, const Params &p) {
    const __fp16 *o = (const __fp16 *)offset.contents;
    const __fp16 *s = (const __fp16 *)scale.contents;
    std::vector<float> result(normalized.size());
    for (uint32_t h=0; h<p.kv_heads; ++h)
        for (uint32_t t=0; t<p.tokens; ++t)
            for (uint32_t c=0; c<p.dim; ++c) {
                NSUInteger i=(NSUInteger(h)*p.tokens+t)*p.dim+c;
                NSUInteger j=normalization==1 ? NSUInteger(h)*p.tokens+t
                    : (NSUInteger(h)*p.pages+t/p.page_tokens)*p.dim+c;
                result[i]=std::fma(normalized[i],float(s[j]),float(o[j]));
                require(std::isfinite(result[i]), "Nonfinite restored tensor value");
            }
    return result;
}
static id<MTLBuffer> halfBuffer(id<MTLDevice> device,const std::vector<float> &values,
                                NSString *label) {
    id<MTLBuffer> result=bufferWithSize(device,values.size()*sizeof(__fp16),label);
    __fp16 *h=(__fp16 *)result.contents;
    for (NSUInteger i=0;i<values.size();++i) h[i]=(__fp16)values[i];
    return result;
}
static std::vector<float> halfValues(id<MTLBuffer> buffer) {
    const __fp16 *h=(const __fp16 *)buffer.contents;
    std::vector<float> result(buffer.length/sizeof(__fp16));
    for (NSUInteger i=0;i<result.size();++i) result[i]=float(h[i]);
    return result;
}
static void writeFloats(NSString *directory, NSString *name, const std::vector<float> &values) {
    NSString *path=[directory stringByAppendingPathComponent:name];
    NSData *data=[NSData dataWithBytes:values.data() length:values.size()*sizeof(float)];
    require([data writeToFile:path atomically:YES],"Cannot write diagnostic readback: "+str(path));
}
static id<MTLTexture> ordinaryTexture(id<MTLDevice> device,id<MTLCommandQueue> queue,
                                      const std::vector<float> &norm,const Params &p,
                                      NSString *label) {
    MTLPixelFormat format=p.components==1?MTLPixelFormatR16Float:MTLPixelFormatRGBA16Float;
    MTLTextureDescriptor *desc=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format
                                                                                width:p.width height:p.height mipmapped:NO];
    desc.textureType=MTLTextureType2DArray;
    desc.arrayLength=p.kv_heads;
    desc.storageMode=MTLStorageModePrivate;
    desc.usage=MTLTextureUsageShaderRead;
    id<MTLTexture> texture=[device newTextureWithDescriptor:desc];
    require(texture!=nil,"Cannot allocate ordinary diagnostic texture");
    texture.label=label;
    NSUInteger rowBytes=alignUp(NSUInteger(p.width)*p.components*2,256);
    NSUInteger sliceBytes=rowBytes*p.height;
    id<MTLBuffer> staging=bufferWithSize(device,sliceBytes*p.kv_heads,@"ordinary texture staging");
    std::memset(staging.contents,0,staging.length);
    for(uint32_t h=0;h<p.kv_heads;++h) for(uint32_t t=0;t<p.tokens;++t) {
        __fp16 *row=(__fp16 *)((uint8_t *)staging.contents+h*sliceBytes+t*rowBytes);
        for(uint32_t c=0;c<p.dim;++c) {
            float value=norm[(NSUInteger(h)*p.tokens+t)*p.dim+c];
            if(p.components==4) row[c]=(__fp16)value;
            else row[c]=(__fp16)value;
        }
    }
    id<MTLCommandBuffer> command=[queue commandBuffer];
    id<MTLBlitCommandEncoder> blit=[command blitCommandEncoder];
    for(uint32_t h=0;h<p.kv_heads;++h)
        [blit copyFromBuffer:staging sourceOffset:h*sliceBytes sourceBytesPerRow:rowBytes
         sourceBytesPerImage:0 sourceSize:MTLSizeMake(p.width,p.height,1) toTexture:texture
           destinationSlice:h destinationLevel:0 destinationOrigin:MTLOriginMake(0,0,0)];
    [blit endEncoding];
    complete(command);
    return texture;
}
static std::vector<float> reference(const float *q,const std::vector<float> &k,
                                    const std::vector<float> &v,const Params &p) {
    std::vector<float> out(NSUInteger(p.query_heads)*p.dim);
    std::vector<double> scores(p.tokens);
    for(uint32_t h=0;h<p.query_heads;++h) {
        uint32_t kh=h/(p.query_heads/p.kv_heads);
        double maximum=-INFINITY;
        for(uint32_t t=0;t<p.tokens;++t) {
            double score=0;
            for(uint32_t c=0;c<p.dim;++c)
                score+=double(q[h*p.dim+c])*k[(NSUInteger(kh)*p.tokens+t)*p.dim+c];
            scores[t]=score/std::sqrt(double(p.dim));
            maximum=std::max(maximum,scores[t]);
        }
        double denominator=0;
        for(double &score:scores) { score=std::exp(score-maximum); denominator+=score; }
        for(uint32_t c=0;c<p.dim;++c) {
            double value=0;
            for(uint32_t t=0;t<p.tokens;++t)
                value+=scores[t]*v[(NSUInteger(kh)*p.tokens+t)*p.dim+c];
            out[h*p.dim+c]=float(value/denominator);
        }
    }
    return out;
}
static NSDictionary *errors(const float *actual,const std::vector<float> &expected) {
    double squared=0,norm=0,maxabs=0;
    bool finite=true;
    for(NSUInteger i=0;i<expected.size();++i) {
        double delta=double(actual[i])-expected[i];
        finite=finite && std::isfinite(actual[i]) && std::isfinite(expected[i]);
        squared+=delta*delta; norm+=double(expected[i])*expected[i];
        maxabs=std::max(maxabs,std::abs(delta));
    }
    if(!finite || !std::isfinite(squared) || !std::isfinite(norm)) return @{@"finite":@NO};
    return @{@"finite":@YES,@"max_abs":@(maxabs),
             @"rmse":@(std::sqrt(squared/std::max(NSUInteger(1),expected.size()))),
             @"relative_l2":@(std::sqrt(squared/std::max(1e-30,norm)))};
}
static NSDictionary *environment(id<MTLDevice> device) {
    NSMutableArray *sets=[NSMutableArray new];
    for(id<MTLCounterSet> set in device.counterSets) {
        NSMutableArray *counters=[NSMutableArray new];
        for(id<MTLCounter> counter in set.counters) [counters addObject:counter.name];
        [sets addObject:@{@"name":set.name,@"counters":counters}];
    }
    NSMutableDictionary *formats=[NSMutableDictionary new];
    for(uint32_t b: {4u,6u,8u}) {
        auto desc=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:astcFormat(b,b)
                                                                    width:b height:b mipmapped:NO];
        desc.storageMode=MTLStorageModePrivate;
        desc.usage=MTLTextureUsageShaderRead;
        id<MTLTexture> texture=[device newTextureWithDescriptor:desc];
        formats[[NSString stringWithFormat:@"ASTC_%ux%u_LDR",b,b]]=@(texture!=nil);
    }
    return @{@"device":device.name,@"registry_id":@(device.registryID),
             @"unified_memory":@(device.hasUnifiedMemory),
             @"physical_memory_bytes":@([NSProcessInfo processInfo].physicalMemory),
             @"os":[NSProcessInfo processInfo].operatingSystemVersionString,
             @"recommended_max_working_set_bytes":@(device.recommendedMaxWorkingSetSize),
             @"max_buffer_length":@(device.maxBufferLength),
             @"supports_apple7":@([device supportsFamily:MTLGPUFamilyApple7]),
             @"supports_apple8":@([device supportsFamily:MTLGPUFamilyApple8]),
             @"astc_allocation_probe":formats,@"counter_sets":sets,
             @"counter_sampling_dispatch":@([device supportsCounterSampling:MTLCounterSamplingPointAtDispatchBoundary]),
             @"note":@"Allocation is not a decoding test; full probe execution tests texture reads."};
}
static void printJSON(id value) {
    NSError *error=nil;
    NSData *data=[NSJSONSerialization dataWithJSONObject:value options:NSJSONWritingPrettyPrinted|NSJSONWritingSortedKeys error:&error];
    require(data!=nil,"JSON serialization failed: "+str(error.localizedDescription));
    std::fwrite(data.bytes,1,data.length,stdout);
    std::fputc('\n',stdout);
}


struct PoolSlot {
    id<MTLBuffer> kHalf, vHalf, vo, vs;
    id<MTLTexture> vTexture;
};
static NSString *sha256(const void *bytes, NSUInteger length) {
    require(length<=std::numeric_limits<CC_LONG>::max(),"Hash input exceeds CC_SHA256 length");
    unsigned char digest[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256(bytes,(CC_LONG)length,digest);
    NSMutableString *hex=[NSMutableString new];
    for(unsigned char b:digest) [hex appendFormat:@"%02x",b];
    return hex;
}
static id<MTLBuffer> cloneBuffer(id<MTLDevice> device,id<MTLBuffer> source,NSString *label) {
    id<MTLBuffer> clone=[device newBufferWithBytes:source.contents length:source.length
                                         options:MTLResourceStorageModeShared];
    require(clone!=nil,"Cannot allocate "+str(label));
    clone.label=label;
    require(clone!=source && clone.contents!=source.contents,"Pool buffers alias source");
    return clone;
}
static id<MTLTexture> cloneTextureAllocation(id<MTLDevice> device,id<MTLTexture> source,NSString *label) {
    auto desc=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:source.pixelFormat
                                                width:source.width height:source.height mipmapped:NO];
    desc.textureType=MTLTextureType2DArray;
    desc.arrayLength=source.arrayLength;
    desc.storageMode=MTLStorageModePrivate;
    desc.usage=MTLTextureUsageShaderRead;
    id<MTLTexture> clone=[device newTextureWithDescriptor:desc];
    require(clone!=nil && clone!=source,"Cannot allocate distinct pool ASTC texture");
    clone.label=label;
    return clone;
}
static NSString *texturePayloadHash(id<MTLCommandQueue> queue,
                                    id<MTLTexture> texture,const ASTCHeader &h,
                                    id<MTLBuffer> readback) {
    @autoreleasepool {
    NSUInteger rowBytes=alignUp(h.rowBytes,256),sliceBytes=rowBytes*h.blockRows;
    require(readback.length>=sliceBytes*texture.arrayLength,"Texture readback buffer too small");
    id<MTLCommandBuffer> command=[queue commandBuffer];
    id<MTLBlitCommandEncoder> blit=[command blitCommandEncoder];
    for(NSUInteger head=0;head<texture.arrayLength;++head)
        [blit copyFromTexture:texture sourceSlice:head sourceLevel:0 sourceOrigin:MTLOriginMake(0,0,0)
                   sourceSize:MTLSizeMake(h.width,h.height,1) toBuffer:readback destinationOffset:head*sliceBytes
        destinationBytesPerRow:rowBytes destinationBytesPerImage:0];
    [blit endEncoding]; complete(command);
    // Hash only payload block rows; driver padding is intentionally excluded.
    std::vector<uint8_t> packed(h.payloadBytes*texture.arrayLength);
    for(NSUInteger head=0;head<texture.arrayLength;++head)
        for(NSUInteger row=0;row<h.blockRows;++row)
            std::memcpy(packed.data()+head*h.payloadBytes+row*h.rowBytes,
                        (const uint8_t *)readback.contents+head*sliceBytes+row*rowBytes,h.rowBytes);
    return sha256(packed.data(),packed.size());
    }
}

int main(int argc,const char **argv) {
    @autoreleasepool {
        try {
            require(argc>=2,"Usage: astc-kv-probe-pool-v5 --environment | MANIFEST METALLIB_OR_METAL_SOURCE [rounds=30] [iterations=pool_size] [tile_tokens=256] [readback_directory|-] [seed=20261028] [pool_size=1]");
            id<MTLDevice> device=MTLCreateSystemDefaultDevice();
            require(device!=nil,"Metal device is unavailable");
            if(std::string(argv[1])=="--environment") { printJSON(environment(device)); return 0; }
            require(argc>=3,"Pass manifest and compiled .metallib paths");
            require([device supportsFamily:MTLGPUFamilyApple7],"Run this probe on Apple Silicon");
            unsigned rounds=argc>3?unsigned(std::stoul(argv[3])):30;
            std::string poolArgument=argc>8?argv[8]:"1";
            require(!poolArgument.empty() && poolArgument.find_first_not_of("0123456789")==std::string::npos,
                    "pool_size must be an unsigned decimal integer");
            const unsigned long parsedPoolSize=std::stoul(poolArgument);
            require(parsedPoolSize>=1 && parsedPoolSize<=64,"pool_size must be in [1,64]");
            unsigned poolSize=unsigned(parsedPoolSize);
            unsigned iterations=argc>4?unsigned(std::stoul(argv[4])):poolSize;
            require(poolSize>=1 && poolSize<=64,"pool_size must be in [1,64]");
            require(iterations==poolSize,"iterations must equal pool_size: one full cycle per batch");
            unsigned tile=argc>5?unsigned(std::stoul(argv[5])):256;
            unsigned seed=argc>7?unsigned(std::stoul(argv[7])):20261028;
            require(rounds>0&&iterations>0&&tile>0,"Rounds, iterations, and tile size must be positive");
            auto totalStart=Clock::now();
            NSString *manifestPath=[NSString stringWithUTF8String:argv[1]];
            NSString *base=[manifestPath stringByDeletingLastPathComponent];
            NSError *error=nil;
            NSData *json=[NSData dataWithContentsOfFile:manifestPath];
            require(json!=nil,"Cannot read manifest");
            NSDictionary *m=[NSJSONSerialization JSONObjectWithData:json options:0 error:&error];
            require([m isKindOfClass:[NSDictionary class]],"Manifest is not an object");
            require([m[@"format"] isEqual:@"astc-kv-probe-v1"],"Unsupported manifest format");
            require([m[@"metadata_dtype"] isEqual:@"float16_le"],"Only float16_le metadata is implemented");
            bool scalar=[m[@"layout"] isEqual:@"scalar"];
            require(scalar||[m[@"layout"] isEqual:@"rgba"],"Unknown tensor layout");
            Params p{};
            p.tokens=[m[@"tokens"] unsignedIntValue]; p.query_heads=[m[@"query_heads"] unsignedIntValue];
            p.kv_heads=[m[@"kv_heads"] unsignedIntValue]; p.dim=[m[@"head_dim"] unsignedIntValue];
            p.page_tokens=[m[@"page_tokens"] unsignedIntValue]; p.components=scalar?1:4;
            auto selectMode=[&](NSString *field,NSString *zero,NSString *one)->uint32_t {
                id value=m[field];
                if(!value || [value isEqual:zero]) return 0;
                require([value isEqual:one],"Unsupported "+str(field));
                return 1;
            };
            p.k_storage=selectMode(@"k_storage",@"astc",@"fp16");
            p.v_storage=selectMode(@"v_storage",@"astc",@"fp16");
            p.k_normalization=selectMode(@"k_normalization",@"page_channel",@"token");
            p.v_normalization=selectMode(@"v_normalization",@"page_channel",@"token");
            require(p.dim==64 && p.k_storage==1 && p.v_storage==0 && p.v_normalization==1,
                    "Pool v5 requires D64, FP16 K and token-normalized ASTC V");
            require(p.tokens>0&&p.query_heads>0&&p.kv_heads>0&&p.page_tokens>0,"Invalid zero shape");
            require(p.query_heads%p.kv_heads==0,"GQA requires query_heads divisible by kv_heads");
            require(p.dim>0&&p.dim<=128&&p.dim%4==0,"Supported head_dim: a positive multiple of 4, at most 128");
            p.pages=(p.tokens+p.page_tokens-1)/p.page_tokens;
            p.tile_tokens=tile; p.splits=(p.tokens+tile-1)/tile;
            id<MTLCommandQueue> queue=[device newCommandQueue];
            require(queue!=nil,"Cannot create command queue");
            require(p.dim==64,"V4 eight-token specialization requires D64");
            auto pipelineStart=Clock::now();
            NSString *libraryPath=[NSString stringWithUTF8String:argv[2]];
            bool sourceCompilation=[libraryPath.pathExtension isEqual:@"metal"];
            id<MTLLibrary> library=nil;
            if(sourceCompilation) {
                NSString *source=[NSString stringWithContentsOfFile:libraryPath
                                                          encoding:NSUTF8StringEncoding error:&error];
                require(source!=nil,"Cannot read Metal source: "+str(error.localizedDescription));
                MTLCompileOptions *options=[MTLCompileOptions new];
                options.languageVersion=MTLLanguageVersion3_0;
                options.fastMathEnabled=NO;
                library=[device newLibraryWithSource:source options:options error:&error];
            } else {
                library=[device newLibraryWithURL:[NSURL fileURLWithPath:libraryPath] error:&error];
            }
            require(library!=nil,"Cannot compile/load Metal library: "+str(error.localizedDescription));
            auto texPipeline=pipeline(device,library,@"attention_split",0);
            auto halfPipeline=pipeline(device,library,@"attention_split",1);
            auto normalizedPipeline=pipeline(device,library,@"attention_split",2);
            auto factoredTexturePipeline=pipeline(device,library,@"attention_split",0,true);
            auto factoredNormalizedPipeline=pipeline(device,library,@"attention_split",2,true);
            auto mergePipeline=pipeline(device,library,@"attention_merge");
            auto decodePipeline=pipeline(device,library,@"decode_texture");
            double pipelineMS=elapsedMS(pipelineStart);
            auto metaElements=[&](uint32_t storage,uint32_t normalization)->NSUInteger {
                if(storage==1) return 0;
                return normalization==1 ? NSUInteger(p.kv_heads)*p.tokens
                    : NSUInteger(p.kv_heads)*p.pages*p.dim;
            };
            NSUInteger kMetaElements=metaElements(p.k_storage,p.k_normalization);
            NSUInteger vMetaElements=metaElements(p.v_storage,p.v_normalization);
            // A shared dummy binding satisfies argument binding validation for
            // inactive metadata inputs. It is not part of logical KV metadata.
            id<MTLBuffer> dummyMetadata=nil;
            if(p.k_storage==1 || p.v_storage==1) {
                dummyMetadata=bufferWithSize(device,16,@"inactive metadata binding");
                std::memset(dummyMetadata.contents,0,dummyMetadata.length);
            }
            auto q=bufferWithData(device,fileData(base,m[@"query_file"],NSUInteger(p.query_heads)*p.dim*4),@"query f32");
            auto ko=kMetaElements ? bufferWithData(device,fileData(base,m[@"k_offset_file"],kMetaElements*2),@"K offset f16") : dummyMetadata;
            auto ks=kMetaElements ? bufferWithData(device,fileData(base,m[@"k_scale_file"],kMetaElements*2),@"K scale f16") : dummyMetadata;
            auto vo=vMetaElements ? bufferWithData(device,fileData(base,m[@"v_offset_file"],vMetaElements*2),@"V offset f16") : dummyMetadata;
            auto vs=vMetaElements ? bufferWithData(device,fileData(base,m[@"v_scale_file"],vMetaElements*2),@"V scale f16") : dummyMetadata;
            for(NSUInteger i=0;i<q.length/4;++i)
                require(std::isfinite(((const float *)q.contents)[i]),"Nonfinite query");
            for(id<MTLBuffer> b in @[ko,ks,vo,vs])
                for(NSUInteger i=0;i<b.length/2;++i)
                    require(std::isfinite(float(((const __fp16 *)b.contents)[i])),"Nonfinite metadata");
            auto uploadStart=Clock::now();
            id<MTLCommandBuffer> upload=[queue commandBuffer];
            upload.label=@"precompressed block upload (NOT encoding)";
            auto blit=[upload blitCommandEncoder];
            Uploaded k{},v{};
            try {
                if(p.k_storage==0) k=uploadASTC(device,blit,base,m[@"k_files"],@"K ASTC",p);
                if(p.v_storage==0) v=uploadASTC(device,blit,base,m[@"v_files"],@"V ASTC",p);
                if(p.k_storage==0 && p.v_storage==0)
                    require(k.header.width==v.header.width&&k.header.height==v.header.height,
                            "K and V atlas dimensions must match when both are ASTC");
            } catch(...) {
                // Invalid manifests may throw after opening the blit encoder.
                // End encoding before ARC destroys it; do not commit partial uploads.
                [blit endEncoding];
                throw;
            }
            p.width=p.k_storage==0?k.header.width:(p.v_storage==0?v.header.width:p.dim/p.components);
            p.height=p.k_storage==0?k.header.height:(p.v_storage==0?v.header.height:p.pages*p.page_tokens);
            [blit endEncoding]; complete(upload);
            double uploadWallMS=elapsedMS(uploadStart);
            id uploadGPUMS=gpuMS(upload);
            NSUInteger compressedStagingBytes=k.stagingBytes+v.stagingBytes;
            k.staging=nil; v.staging=nil;
            std::vector<float> kn,vn,kr,vr;
            id<MTLBuffer> kHalf=nil,vHalf=nil;
            id<MTLTexture> kTex=nil,vTex=nil;
            NSUInteger tensorHalfBytes=NSUInteger(p.kv_heads)*p.tokens*p.dim*2;
            if(p.k_storage==0) {
                kn=decode(device,queue,decodePipeline,k.texture,p);
                kr=restore(kn,ko,ks,p.k_normalization,p);
                kHalf=halfBuffer(device,kr,@"K FP16 diagnostic buffer");
                kTex=ordinaryTexture(device,queue,kn,p,@"K uncompressed diagnostic texture");
            } else {
                kHalf=bufferWithData(device,fileData(base,m[@"k_file"],tensorHalfBytes),@"K original FP16 cache");
                kr=halfValues(kHalf);
            }
            if(p.v_storage==0) {
                vn=decode(device,queue,decodePipeline,v.texture,p);
                vr=restore(vn,vo,vs,p.v_normalization,p);
                vHalf=halfBuffer(device,vr,@"V FP16 diagnostic buffer");
                vTex=ordinaryTexture(device,queue,vn,p,@"V uncompressed diagnostic texture");
            } else {
                vHalf=bufferWithData(device,fileData(base,m[@"v_file"],tensorHalfBytes),@"V original FP16 cache");
                vr=halfValues(vHalf);
            }
            for(float value:kr) require(std::isfinite(value),"Nonfinite K input");
            for(float value:vr) require(std::isfinite(value),"Nonfinite V input");
            if(argc>6 && std::string(argv[6])!="-") {
                NSString *readbackDirectory=[NSString stringWithUTF8String:argv[6]];
                BOOL isDirectory=NO;
                require([[NSFileManager defaultManager] fileExistsAtPath:readbackDirectory isDirectory:&isDirectory] && isDirectory,
                        "Diagnostic readback directory must already exist");
                writeFloats(readbackDirectory,@"k_hardware.f32",kr);
                writeFloats(readbackDirectory,@"v_hardware.f32",vr);
                if(!kn.empty()) writeFloats(readbackDirectory,@"k_normalized_hardware.f32",kn);
                if(!vn.empty()) writeFloats(readbackDirectory,@"v_normalized_hardware.f32",vn);
            }
            // If both sides are FP16, bind one unused texture so the texture
            // specialization still has a valid argument table. No read occurs.
            id<MTLTexture> dummyTexture=nil;
            if(p.k_storage==1 && p.v_storage==1) {
                auto desc=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR16Float
                                                                             width:1 height:1 mipmapped:NO];
                desc.textureType=MTLTextureType2DArray; desc.arrayLength=1;
                desc.storageMode=MTLStorageModePrivate; desc.usage=MTLTextureUsageShaderRead;
                dummyTexture=[device newTextureWithDescriptor:desc];
                require(dummyTexture!=nil,"Cannot allocate inactive texture binding");
            }
            // Separate reference for each precision path prevents conflating
            // codec loss, normalized texture rounding, and FP16 storage error.
            auto kOrdinary=p.k_storage==0?restore(decode(device,queue,decodePipeline,kTex,p),ko,ks,p.k_normalization,p):kr;
            auto vOrdinary=p.v_storage==0?restore(decode(device,queue,decodePipeline,vTex,p),vo,vs,p.v_normalization,p):vr;
            // Same normalized FP16 scalars as the ordinary textures; same metadata
            // and affine4 reconstruction in the shader. Only their load path differs.
            auto kNormalizedHalf=p.k_storage==0?halfBuffer(device,kn,@"K normalized FP16 control"):kHalf;
            auto vNormalizedHalf=p.v_storage==0?halfBuffer(device,vn,@"V normalized FP16 control"):vHalf;
            auto kNormalized=p.k_storage==0?restore(halfValues(kNormalizedHalf),ko,ks,p.k_normalization,p):kr;
            auto vNormalized=p.v_storage==0?restore(halfValues(vNormalizedHalf),vo,vs,p.v_normalization,p):vr;
            // Four channels in one ordinary RGBA16Float texel. This is not an
            // RGBA ASTC representation and introduces no additional codec loss.
            Params rgbaParams=p;
            rgbaParams.components=4; rgbaParams.width=p.dim/4;
            id<MTLTexture> kRGBA=p.k_storage==0?ordinaryTexture(device,queue,kn,rgbaParams,@"K normalized RGBA16 control"):nil;
            id<MTLTexture> vRGBA=p.v_storage==0?ordinaryTexture(device,queue,vn,rgbaParams,@"V normalized RGBA16 control"):nil;
            auto kPacked=p.k_storage==0?restore(decode(device,queue,decodePipeline,kRGBA,rgbaParams),ko,ks,p.k_normalization,p):kr;
            auto vPacked=p.v_storage==0?restore(decode(device,queue,decodePipeline,vRGBA,rgbaParams),vo,vs,p.v_normalization,p):vr;
            auto astcReference=reference((const float *)q.contents,kr,vr,p);
            auto texReference=reference((const float *)q.contents,kOrdinary,vOrdinary,p);
            auto halfReference=reference((const float *)q.contents,halfValues(kHalf),halfValues(vHalf),p);
            auto normalizedReference=reference((const float *)q.contents,kNormalized,vNormalized,p);
            auto packedReference=reference((const float *)q.contents,kPacked,vPacked,p);
            auto partial=bufferWithSize(device,NSUInteger(p.query_heads)*p.splits*p.dim*4,@"partial attention outputs");
            auto stats=bufferWithSize(device,NSUInteger(p.query_heads)*p.splits*8,@"partial softmax states");
            auto output=bufferWithSize(device,NSUInteger(p.query_heads)*p.dim*4,@"attention output");
            auto run=[&](int variant,unsigned count)->NSDictionary * {
                auto begin=Clock::now();
                id<MTLCommandBuffer> cmd=[queue commandBuffer];
                cmd.label=@"attention split + merge; no codec or full decode";
                id<MTLComputeCommandEncoder> enc=[cmd computeCommandEncoder];
                const Params &variantParams=variant==4?rgbaParams:p;
                [enc setBytes:&variantParams length:sizeof(variantParams) atIndex:0];
                [enc setBuffer:q offset:0 atIndex:1];
                [enc setBuffer:((variant==3 || variant==6)?kNormalizedHalf:kHalf) offset:0 atIndex:2];
                [enc setBuffer:((variant==3 || variant==6)?vNormalizedHalf:vHalf) offset:0 atIndex:3];
                [enc setBuffer:ko offset:0 atIndex:4]; [enc setBuffer:ks offset:0 atIndex:5];
                [enc setBuffer:vo offset:0 atIndex:6]; [enc setBuffer:vs offset:0 atIndex:7];
                [enc setBuffer:partial offset:0 atIndex:8]; [enc setBuffer:stats offset:0 atIndex:9];
                [enc setBuffer:output offset:0 atIndex:10];
                id<MTLTexture> boundK=variant==4?kRGBA:(variant==1?kTex:k.texture);
                id<MTLTexture> boundV=variant==4?vRGBA:(variant==1?vTex:v.texture);
                if(!boundK) boundK=boundV?boundV:dummyTexture;
                if(!boundV) boundV=boundK?boundK:dummyTexture;
                [enc setTexture:boundK atIndex:0]; [enc setTexture:boundV atIndex:1];
                for(unsigned j=0;j<count;++j) {
                    [enc setComputePipelineState:(variant==5?factoredTexturePipeline:(variant==6?factoredNormalizedPipeline:(variant==2?halfPipeline:(variant==3?normalizedPipeline:texPipeline))))];
                    [enc dispatchThreadgroups:MTLSizeMake(p.query_heads,p.splits,1)
                         threadsPerThreadgroup:MTLSizeMake(32,1,1)];
                    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
                    [enc setComputePipelineState:mergePipeline];
                    [enc dispatchThreadgroups:MTLSizeMake(p.query_heads,1,1)
                         threadsPerThreadgroup:MTLSizeMake(32,1,1)];
                    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
                }
                [enc endEncoding]; complete(cmd);
                id duration=gpuMS(cmd);
                return @{@"gpu_batch_ms":duration,@"wall_batch_ms":@(elapsedMS(begin)),@"iterations":@(count)};
            };
            NSArray *names=@[@"astc_texture",@"uncompressed_texture",@"fp16_buffer",
                             @"normalized_fp16_buffer",@"uncompressed_rgba_texture",
                             @"astc_factored_v",@"normalized_buffer_factored_v"];
            NSMutableDictionary *correctness=[NSMutableDictionary new];
            bool passed=true;
            const std::vector<float> *references[]={&astcReference,&texReference,&halfReference,
                                                     &normalizedReference,&packedReference,&astcReference,&normalizedReference};
            for(int variant=0;variant<7;++variant) {
                run(variant,1);
                NSDictionary *e=errors((const float *)output.contents,*references[variant]);
                bool good=[e[@"finite"] boolValue]&&[e[@"relative_l2"] doubleValue]<=1e-4;
                correctness[names[variant]]=@{@"vs_same_values_cpu_fp64_accumulation":e,@"passed_rtol_l2_1e_4":@(good)};
                passed=passed&&good;
            }
            for(NSString *prefix in @[@"k",@"v"]) {
                bool isK=[prefix isEqual:@"k"];
                const auto &normalized=isK?kNormalized:vNormalized;
                const auto &ordinary=isK?kOrdinary:vOrdinary;
                const auto &packed=isK?kPacked:vPacked;
                for(int control=0;control<2;++control) {
                    const auto &values=control==0?ordinary:packed;
                    NSDictionary *e=errors(values.data(),normalized);
                    bool good=[e[@"finite"] boolValue] && [e[@"max_abs"] doubleValue]==0.0;
                    NSString *suffix=control==0?@"_ordinary_vs_normalized_buffer":@"_rgba_vs_normalized_buffer";
                    correctness[[prefix stringByAppendingString:suffix]]=@{
                        @"errors":e,@"passed_exact_values":@(good)};
                    passed=passed&&good;
                }
            }
            for(NSString *prefix in @[@"k",@"v"]) {
                NSString *key=[prefix stringByAppendingString:@"_decoded_file"];
                bool isK=[prefix isEqual:@"k"];
                bool compressed=isK?p.k_storage==0:p.v_storage==0;
                if(compressed && !m[key]) {
                    correctness[[prefix stringByAppendingString:@"_decode_gate"]]=@{
                        @"passed":@NO,@"reason":@"Missing offline decode reference"};
                    passed=false;
                }
                if(m[key]) {
                    NSData *data=fileData(base,m[key],NSUInteger(p.kv_heads)*p.tokens*p.dim*4);
                    const auto &hw=[prefix isEqual:@"k"]?kr:vr;
                    std::vector<float> sw((const float *)data.bytes,(const float *)data.bytes+hw.size());
                    NSString *suffix=compressed?@"_gpu_vs_offline_decode":@"_fp16_vs_offline_values";
                    NSDictionary *decodeErrors=errors(hw.data(),sw);
                    correctness[[prefix stringByAppendingString:suffix]]=decodeErrors;
                    bool good=[decodeErrors[@"finite"] boolValue];
                    // Engineering smoke bound calibrated after the first two
                    // fixtures, not a preregistered model-quality tolerance or
                    // a claim that the two decode paths are bit-identical.
                    const __fp16 *scales=(const __fp16 *)(isK?ks:vs).contents;
                    uint32_t normalization=isK?p.k_normalization:p.v_normalization;
                    for(uint32_t h=0;good && h<p.kv_heads;++h)
                        for(uint32_t t=0;good && t<p.tokens;++t)
                            for(uint32_t c=0;good && c<p.dim;++c) {
                                NSUInteger i=(NSUInteger(h)*p.tokens+t)*p.dim+c;
                                NSUInteger j=normalization==1?NSUInteger(h)*p.tokens+t:
                                    (NSUInteger(h)*p.pages+t/p.page_tokens)*p.dim+c;
                                double scale=compressed?double(scales[j]):0.0;
                                if(compressed && (!std::isfinite(scale) || scale<=0)) { good=false;break; }
                                double roundoff=8*std::numeric_limits<float>::epsilon()*
                                    std::max(1.0,std::max(std::abs(double(hw[i])),std::abs(double(sw[i]))));
                                double limit=compressed?scale/1024.0+roundoff:0.0;
                                good=std::abs(double(hw[i])-sw[i])<=limit;
                            }
                    correctness[[prefix stringByAppendingString:@"_decode_gate"]]=@{
                        @"passed":@(good),@"criterion":compressed?@"abs_delta <= scale/1024 + FP32_roundoff":@"exact FP16 values",
                        @"scope":@"engineering fixture check; no model-quality or exact ASTC equivalence claim"};
                    passed=passed&&good;
                }
            }
            if(!passed) printJSON(@{@"schema":@"astc-kv-metal-probe-pool-failure-v5",
                @"status":@"correctness_failed",@"correctness":correctness,@"raw_samples":@[]});
            require(passed,"Attention correctness gate failed; no timing result is published");
            constexpr NSUInteger maximumPoolBytes=NSUInteger(1)<<30;
            const NSUInteger estimatedSlotBytes=kHalf.allocatedSize+vHalf.allocatedSize+
                                                vo.allocatedSize+vs.allocatedSize+v.texture.allocatedSize;
            require(estimatedSlotBytes>0 && estimatedSlotBytes<=maximumPoolBytes/poolSize,
                    "Pool allocation estimate exceeds 1 GiB");
            std::vector<PoolSlot> pool;
            pool.reserve(poolSize);
            NSMutableArray *slotMemory=[NSMutableArray new],*slotHashes=[NSMutableArray new];
            NSMutableSet *resourceIdentities=[NSMutableSet new];
            NSUInteger poolAllocated=0,poolK=0,poolVHalf=0,poolVASTC=0,poolMetadata=0;
            auto poolStart=Clock::now();
            for(unsigned i=0;i<poolSize;++i) {
                PoolSlot slot{cloneBuffer(device,kHalf,@"pool K FP16"),cloneBuffer(device,vHalf,@"pool V FP16"),
                              cloneBuffer(device,vo,@"pool V offset"),cloneBuffer(device,vs,@"pool V scale"),
                              cloneTextureAllocation(device,v.texture,@"pool V ASTC")};
                const NSUInteger slotBytes=slot.kHalf.allocatedSize+slot.vHalf.allocatedSize+
                    slot.vo.allocatedSize+slot.vs.allocatedSize+slot.vTexture.allocatedSize;
                require(slotBytes<=maximumPoolBytes-poolAllocated,"Actual pool allocation exceeds 1 GiB");
                poolAllocated+=slotBytes;
                poolK+=slot.kHalf.allocatedSize; poolVHalf+=slot.vHalf.allocatedSize;
                poolVASTC+=slot.vTexture.allocatedSize; poolMetadata+=slot.vo.allocatedSize+slot.vs.allocatedSize;
                for(id resource in @[slot.kHalf,slot.vHalf,slot.vo,slot.vs,slot.vTexture]) {
                    NSValue *identity=[NSValue valueWithPointer:(__bridge const void *)resource];
                    require(![resourceIdentities containsObject:identity],"Pool slot resource object alias");
                    [resourceIdentities addObject:identity];
                }
                [slotMemory addObject:@{@"slot":@(i),@"k_half_allocated_bytes":@(slot.kHalf.allocatedSize),
                    @"v_half_allocated_bytes":@(slot.vHalf.allocatedSize),@"v_astc_allocated_bytes":@(slot.vTexture.allocatedSize),
                    @"v_offset_allocated_bytes":@(slot.vo.allocatedSize),@"v_scale_allocated_bytes":@(slot.vs.allocatedSize),
                    @"total_allocated_bytes":@(slotBytes)}];
                pool.push_back(slot);
            }
            // All full-slice copies finish before verification/warmup/timing.
            id<MTLCommandBuffer> poolCopy=[queue commandBuffer];
            auto poolBlit=[poolCopy blitCommandEncoder];
            for(const auto &slot:pool) for(NSUInteger head=0;head<v.texture.arrayLength;++head)
                [poolBlit copyFromTexture:v.texture sourceSlice:head sourceLevel:0 sourceOrigin:MTLOriginMake(0,0,0)
                             sourceSize:MTLSizeMake(v.texture.width,v.texture.height,1) toTexture:slot.vTexture
                       destinationSlice:head destinationLevel:0 destinationOrigin:MTLOriginMake(0,0,0)];
            [poolBlit endEncoding]; complete(poolCopy);
            const double poolCopyWallMS=elapsedMS(poolStart);
            const NSUInteger verificationBytes=alignUp(v.header.rowBytes,256)*v.header.blockRows*p.kv_heads;
            auto textureVerification=bufferWithSize(device,verificationBytes,@"pool ASTC raw-byte verification");
            NSString *sourceTextureHash=texturePayloadHash(queue,v.texture,v.header,textureVerification);
            std::vector<uint8_t> expectedPayload;
            for(id file in m[@"v_files"]) {
                NSData *image=fileData(base,file);
                const uint8_t *bytes=(const uint8_t *)image.bytes;
                expectedPayload.insert(expectedPayload.end(),bytes+16,bytes+image.length);
            }
            require([sourceTextureHash isEqual:sha256(expectedPayload.data(),expectedPayload.size())],
                    "Source ASTC texture bytes differ from file payload");
            const std::array<id<MTLBuffer>,4> originals={kHalf,vHalf,vo,vs};
            const std::array<NSString *,4> hashKeys={@"k_half_sha256",@"v_half_sha256",@"v_offset_sha256",@"v_scale_sha256"};
            for(unsigned i=0;i<poolSize;++i) {
                const auto &slot=pool[i];
                const std::array<id<MTLBuffer>,4> copies={slot.kHalf,slot.vHalf,slot.vo,slot.vs};
                NSMutableDictionary *hashes=[@{@"slot":@(i)} mutableCopy];
                for(unsigned b=0;b<copies.size();++b) {
                    NSString *copyHash=sha256(copies[b].contents,copies[b].length);
                    require([copyHash isEqual:sha256(originals[b].contents,originals[b].length)],"Pool CPU buffer hash mismatch");
                    hashes[hashKeys[b]]=copyHash;
                }
                NSString *textureHash=texturePayloadHash(queue,slot.vTexture,v.header,textureVerification);
                require([textureHash isEqual:sourceTextureHash],"Pool ASTC texture payload hash mismatch");
                hashes[@"v_astc_payload_sha256"]=textureHash;
                hashes[@"passed"]=@YES;
                [slotHashes addObject:hashes];
            }
            const NSUInteger verificationAllocated=textureVerification.allocatedSize;
            textureVerification=nil;
            std::array<uint64_t,7> cursors{};
            auto runPool=[&](int variant,unsigned count)->NSDictionary * {
                require(variant==5 || variant==2,"Only ASTC factored and FP16 are timed in pool probe");
                const uint64_t startCursor=cursors[variant];
                auto begin=Clock::now();
                id<MTLCommandBuffer> cmd=[queue commandBuffer];
                cmd.label=@"v5 resident pool attention split + merge";
                auto enc=[cmd computeCommandEncoder];
                [enc setBytes:&p length:sizeof(p) atIndex:0];
                [enc setBuffer:q offset:0 atIndex:1];
                [enc setBuffer:ko offset:0 atIndex:4]; [enc setBuffer:ks offset:0 atIndex:5];
                [enc setBuffer:partial offset:0 atIndex:8]; [enc setBuffer:stats offset:0 atIndex:9];
                [enc setBuffer:output offset:0 atIndex:10];
                for(unsigned j=0;j<count;++j) {
                    const auto &slot=pool[cursors[variant]++%poolSize];
                    // Identical binding calls on both paths; only the pipeline differs.
                    [enc setBuffer:slot.kHalf offset:0 atIndex:2];
                    [enc setBuffer:slot.vHalf offset:0 atIndex:3];
                    [enc setBuffer:slot.vo offset:0 atIndex:6]; [enc setBuffer:slot.vs offset:0 atIndex:7];
                    [enc setTexture:slot.vTexture atIndex:0]; [enc setTexture:slot.vTexture atIndex:1];
                    [enc setComputePipelineState:variant==5?factoredTexturePipeline:halfPipeline];
                    [enc dispatchThreadgroups:MTLSizeMake(p.query_heads,p.splits,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
                    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
                    [enc setComputePipelineState:mergePipeline];
                    [enc dispatchThreadgroups:MTLSizeMake(p.query_heads,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
                    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
                }
                [enc endEncoding]; complete(cmd);
                return @{@"gpu_batch_ms":gpuMS(cmd),@"wall_batch_ms":@(elapsedMS(begin)),@"iterations":@(count),
                    @"pool_cursor_before":@(startCursor),@"first_slot":@(startCursor%poolSize),
                    @"last_slot":@((cursors[variant]-1)%poolSize)};
            };
            NSMutableArray *poolCorrectness=[NSMutableArray new];
            bool poolPassed=true;
            for(int variant:{5,2}) {
                auto checkOutput=[&](NSString *phase,unsigned count) {
                    NSDictionary *dispatch=runPool(variant,count);
                    NSDictionary *e=errors((const float *)output.contents,*references[variant]);
                    bool good=[e[@"finite"] boolValue] && [e[@"relative_l2"] doubleValue]<=1e-4;
                    [poolCorrectness addObject:@{@"variant":names[variant],@"phase":phase,
                        @"slot":dispatch[@"last_slot"],@"cursor_before":dispatch[@"pool_cursor_before"],
                        @"iterations":@(count),@"errors":e,@"passed_rtol_l2_1e_4":@(good)}];
                    poolPassed=poolPassed&&good;
                };
                for(unsigned slot=0;slot<poolSize;++slot) checkOutput(@"every_slot",1);
                checkOutput(@"wrap_first",1);
                if(poolSize>1) checkOutput(@"wrap_last",poolSize-1);
            }
            if(!poolPassed) printJSON(@{@"schema":@"astc-kv-metal-probe-pool-failure-v5",@"status":@"correctness_failed",
                @"correctness":correctness,@"pool_correctness":poolCorrectness,@"raw_samples":@[]});
            require(poolPassed,"Pool correctness failed; no timing result is published");
            cursors.fill(0);
            std::mt19937 rng(seed);
            std::array<int,2> order={5,2};
            std::shuffle(order.begin(),order.end(),rng);
            for(int variant:order) runPool(variant,poolSize); // one complete warmup cycle
            NSMutableArray *samples=[NSMutableArray new];
            for(unsigned round=0;round<rounds;++round) {
                std::shuffle(order.begin(),order.end(),rng);
                for(int index=0;index<2;++index) {
                    int variant=order[index];
                    NSMutableDictionary *sample=[runPool(variant,iterations) mutableCopy];
                    sample[@"round"]=@(round); sample[@"variant"]=names[variant];
                    sample[@"order_within_round"]=@(index); [samples addObject:sample];
                }
            }
            struct rusage usage{}; getrusage(RUSAGE_SELF,&usage);
            NSUInteger metadataBytes=(kMetaElements+vMetaElements)*4;
            NSUInteger metadataAllocated=(p.k_storage==0?ko.allocatedSize+ks.allocatedSize:0)
                                        +(p.v_storage==0?vo.allocatedSize+vs.allocatedSize:0);
            NSUInteger payloadBytes=k.payloadBytes+v.payloadBytes;
            NSUInteger fp16CacheBytes=(p.k_storage==1?kHalf.length:0)+(p.v_storage==1?vHalf.length:0);
            NSUInteger fp16CacheAllocated=(p.k_storage==1?kHalf.allocatedSize:0)+(p.v_storage==1?vHalf.allocatedSize:0);
            NSUInteger kvScalars=NSUInteger(2)*p.kv_heads*p.tokens*p.dim;
            NSUInteger scratchBytes=partial.allocatedSize+stats.allocatedSize+output.allocatedSize;
            // V-only restriction makes this a unique-resource sum (K controls alias kHalf).
            NSUInteger nonpoolResources=kHalf.allocatedSize+vHalf.allocatedSize+vo.allocatedSize+vs.allocatedSize+
                v.texture.allocatedSize+vTex.allocatedSize+vNormalizedHalf.allocatedSize+vRGBA.allocatedSize+
                dummyMetadata.allocatedSize+q.allocatedSize+scratchBytes;
            NSMutableDictionary *result=[@{
                @"schema":@"astc-kv-metal-probe-pool-result-v5",@"status":@"executed_on_metal",
                @"hardware_astc_read_exercised":@(p.k_storage==0 || p.v_storage==0),
                @"scope":@"precompressed, one-query diagnostic; no runtime speedup claim",
                @"shader_compilation":sourceCompilation?@"runtime_source_metal3_no_fast_math":@"offline_metallib",
                @"environment":environment(device),@"input_manifest":m,
                @"shape":@{@"tokens":@(p.tokens),@"head_dim":@(p.dim),@"query_heads":@(p.query_heads),
                    @"kv_heads":@(p.kv_heads),@"page_tokens":@(p.page_tokens),@"tile_tokens":@(p.tile_tokens),
                    @"splits":@(p.splits),@"atlas_width":@(p.width),@"atlas_height":@(p.height)},
                @"correctness":correctness,@"raw_samples":samples,
                @"pool_correctness":poolCorrectness,@"pool_slot_hashes":slotHashes,
                @"pool":@{@"size":@(poolSize),@"distinct_allocations":@YES,@"identical_values":@YES,
                    @"maximum_allocated_bytes":@(maximumPoolBytes),@"slots":slotMemory,
                    @"query_sha256":sha256(q.contents,q.length),@"source_astc_payload_sha256":sourceTextureHash,
                    @"slot_selection":@"per-variant cursor modulo pool_size, bindings inside each iteration",
                    @"bind_calls_identical_across_variants":@YES,
                    @"pool_copy_wall_ms":@(poolCopyWallMS),@"pool_copy_gpu_ms":gpuMS(poolCopy)},
                @"attribution_controls":@{
                    @"factored_v_applied":@(p.v_storage==0 && p.v_normalization==1),
                    @"factored_v_identity":@"sum(p*(s*u+o)) = sum((p*s)*u) + sum(p*o); per-token V only",
                    @"normalized_fp16_buffer":@"Same FP16 normalized scalars, metadata and affine reconstruction as ordinary textures",
                    @"uncompressed_rgba_texture":@"Ordinary RGBA16Float packing of the same normalized scalars; not RGBA ASTC",
                    @"ordinary_input_components":@(p.components),
                    @"packed_texture_width":@(rgbaParams.width),
                    @"scalar_vs_rgba_attribution_applicable":@(p.components==1),
                    @"same_query_buffer_all_variants":@YES,
                    @"schedule":@"eight tokens per SIMD group wave; four lanes per token; D64 only"},
                @"timing":@{@"pipeline_creation_wall_ms":@(pipelineMS),
                    @"block_upload_wall_ms":@(uploadWallMS),@"block_upload_gpu_ms":uploadGPUMS,
                    @"process_elapsed_ms":@(elapsedMS(totalStart)),@"codec_time_ms":[NSNull null],
                    @"decode_or_model_latency_ms":[NSNull null],@"warmup_iterations_per_variant":@(poolSize),
                    @"warmup_command_buffers_per_variant":@1,
                    @"variant_order":@"seeded shuffle each warmup and measured round",
                    @"order_seed":@(seed),@"rounds":@(rounds),@"iterations_per_round":@(iterations),
                    @"cache_state":@"distinct resident allocations; one full pool cycle per batch; no forced eviction or measured cache miss rate"},
                @"memory":@{@"astc_file_bytes":@(k.fileBytes+v.fileBytes),
                    @"pool_total_allocated_bytes":@(poolAllocated),
                    @"pool_k_shared_between_paths_allocated_bytes":@(poolK),
                    @"pool_v_fp16_allocated_bytes":@(poolVHalf),
                    @"pool_v_astc_allocated_bytes":@(poolVASTC),
                    @"pool_v_metadata_allocated_bytes":@(poolMetadata),
                    @"pool_astc_shader_read_working_set_allocated_bytes":@(poolK+poolVASTC+poolMetadata),
                    @"pool_fp16_shader_read_working_set_allocated_bytes":@(poolK+poolVHalf),
                    @"pool_astc_with_shared_query_scratch_allocated_bytes":@(poolK+poolVASTC+poolMetadata+q.allocatedSize+scratchBytes),
                    @"pool_fp16_with_shared_query_scratch_allocated_bytes":@(poolK+poolVHalf+q.allocatedSize+scratchBytes),
                    @"pool_resource_counts":@{@"k_buffers":@(poolSize),@"v_fp16_buffers":@(poolSize),
                        @"v_astc_textures":@(poolSize),@"v_metadata_buffers":@(2*poolSize),
                        @"shared_query_buffers":@1,@"shared_partial_stats_output_buffers":@3},
                    @"pool_texture_verification_transient_allocated_bytes":@(verificationAllocated),
                    @"original_nonpool_unique_gpu_resources_allocated_bytes":@(nonpoolResources),
                    @"pool_plus_original_unique_gpu_resources_allocated_bytes":@(poolAllocated+nonpoolResources),
                    @"original_nonpool_unique_resource_counts":@{@"buffers":@10,@"textures":@3},
                    @"shared_query_allocated_bytes":@(q.allocatedSize),
                    @"binding_note":@"Both paths bind all slot resources identically; read working set counts only shader-accessed K/V/metadata. All diagnostic resources remain resident.",
                    @"astc_payload_with_texture_padding_bytes":@(payloadBytes),
                    @"fp16_cache_logical_bytes":@(fp16CacheBytes),
                    @"metadata_logical_bytes":@(metadataBytes),
                    @"logical_effective_bits_per_original_kv_scalar":@(8.0*(payloadBytes+fp16CacheBytes+metadataBytes)/kvScalars),
                    @"k_logical_effective_bits_per_scalar":@(16.0*(k.payloadBytes+(p.k_storage==1?kHalf.length:0)+4*kMetaElements)/kvScalars),
                    @"v_logical_effective_bits_per_scalar":@(16.0*(v.payloadBytes+(p.v_storage==1?vHalf.length:0)+4*vMetaElements)/kvScalars),
                    @"astc_texture_allocated_bytes":@(k.texture.allocatedSize+v.texture.allocatedSize),
                    @"fp16_cache_allocated_bytes":@(fp16CacheAllocated),
                    @"metadata_allocated_bytes":@(metadataAllocated),
                    @"control_params_bytes":@(sizeof(Params)),
                    @"inactive_binding_allocated_bytes":@(dummyMetadata.allocatedSize+dummyTexture.allocatedSize),
                    @"compressed_upload_staging_bytes":@(compressedStagingBytes),
                    @"attention_scratch_allocated_bytes":@(scratchBytes),
                    @"fp16_buffer_allocated_bytes":@(kHalf.allocatedSize+vHalf.allocatedSize),
                    @"fp16_decode_control_only_allocated_bytes":@((p.k_storage==0?kHalf.allocatedSize:0)+(p.v_storage==0?vHalf.allocatedSize:0)),
                    @"ordinary_texture_allocated_bytes":@(kTex.allocatedSize+vTex.allocatedSize),
                    @"normalized_buffer_control_allocated_bytes":@((p.k_storage==0?kNormalizedHalf.allocatedSize:0)+(p.v_storage==0?vNormalizedHalf.allocatedSize:0)),
                    @"ordinary_rgba_control_allocated_bytes":@(kRGBA.allocatedSize+vRGBA.allocatedSize),
                    @"device_current_allocated_bytes_all_diagnostics":@(device.currentAllocatedSize),
                    @"process_max_rss_bytes_macos":@(usage.ru_maxrss),
                    @"production_peak_bytes":[NSNull null],
                    @"peak_note":@"Pool plus original seven-path control resources and decoded CPU tensors coexist. Read working sets do not imply exclusive residency, cache misses or DRAM bandwidth. Process peak is diagnostic, not production."}
            } mutableCopy];
            printJSON(result); return 0;
        } catch(const std::exception &e) {
            std::fprintf(stderr,"astc-kv-probe: %s\n",e.what()); return 1;
        }
    }
}
