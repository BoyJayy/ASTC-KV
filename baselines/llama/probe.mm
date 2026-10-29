#import <Foundation/Foundation.h>
#include <CommonCrypto/CommonDigest.h>
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-metal.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using Clock=std::chrono::steady_clock;
static void require(bool ok,const std::string &why) { if(!ok) throw std::runtime_error(why); }
static NSString *ns(const char *s) { return s?[NSString stringWithUTF8String:s]:@""; }
static double ms(Clock::time_point start) { return std::chrono::duration<double,std::milli>(Clock::now()-start).count(); }
static NSString *hashBytes(const void *data,size_t n) {
    require(n<=UINT32_MAX,"Hash input exceeds bounded fixture size");
    unsigned char digest[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256(data,(CC_LONG)n,digest);
    char hex[2*CC_SHA256_DIGEST_LENGTH+1];
    for(size_t i=0;i<sizeof(digest);++i) std::snprintf(hex+2*i,3,"%02x",digest[i]);
    return ns(hex);
}
static void printJSON(id obj) {
    NSError *e=nil;
    NSData *data=[NSJSONSerialization dataWithJSONObject:obj options:NSJSONWritingPrettyPrinted|NSJSONWritingSortedKeys error:&e];
    require(data!=nil,"Cannot serialize finite JSON result");
    std::fwrite(data.bytes,1,data.length,stdout);std::fputc('\n',stdout);
}
static NSData *readBounded(NSString *path,size_t expected) {
    NSDictionary *attrs=[[NSFileManager defaultManager] attributesOfItemAtPath:path error:nil];
    require(attrs!=nil && [attrs fileSize]==expected,"Wrong or missing input byte length: "+std::string(path.UTF8String));
    NSData *data=[NSData dataWithContentsOfFile:path];
    require(data!=nil && data.length==expected,"Cannot read input");
    return data;
}
static NSData *fixtureData(NSString *base,id relative,size_t expected) {
    require([relative isKindOfClass:[NSString class]] && ![(NSString *)relative isAbsolutePath],"Expected relative fixture path");
    NSString *path=[[[base stringByAppendingPathComponent:relative] stringByStandardizingPath] stringByResolvingSymlinksInPath];
    require([path hasPrefix:[base stringByAppendingString:@"/"]],"Fixture path escapes manifest directory");
    return readBounded(path,expected);
}
struct Input {
    int64_t t=0,hq=0,hkv=0,d=0;
    std::vector<float> q,k,v;
    NSDictionary *manifest=nil;
    NSDictionary *provenance=nil;
};
static void checkFinite(const std::vector<float> &v) {
    for(float x:v) require(std::isfinite(x),"Nonfinite input or represented scalar");
}
static Input loadInput(NSString *manifestPath) {
    Input in;
    NSString *absolute=[[manifestPath stringByStandardizingPath] stringByResolvingSymlinksInPath];
    if(!absolute.isAbsolutePath) absolute=[[[NSFileManager defaultManager] currentDirectoryPath] stringByAppendingPathComponent:absolute];
    NSString *base=[absolute stringByDeletingLastPathComponent];
    NSDictionary *attrs=[[NSFileManager defaultManager] attributesOfItemAtPath:absolute error:nil];
    require(attrs && [attrs fileSize]<1024*1024,"Missing/oversized manifest");
    NSData *raw=readBounded(absolute,[attrs fileSize]);
    id parsed=[NSJSONSerialization JSONObjectWithData:raw options:0 error:nil];
    require([parsed isKindOfClass:[NSDictionary class]],"Manifest is not an object");
    in.manifest=parsed;
    require([in.manifest[@"format"] isEqual:@"astc-kv-probe-v1"],"Unknown fixture format");
    auto dim=[&](NSString *key,int64_t maximum) {
        id value=in.manifest[key];
        require([value isKindOfClass:[NSNumber class]],"Expected numeric shape");
        double x=[value doubleValue];
        require(std::isfinite(x)&&x>=1&&x<=maximum&&std::floor(x)==x,"Invalid bounded shape");
        return int64_t(x);
    };
    in.t=dim(@"tokens",32768);in.hq=dim(@"query_heads",256);in.hkv=dim(@"kv_heads",256);in.d=dim(@"head_dim",128);
    require((in.d==64||in.d==128)&&in.hq%in.hkv==0,"Require D64/128 and valid GQA");
    size_t count=size_t(in.t)*in.hkv*in.d;
    require(count<=16*1024*1024,"Fixture scalar count exceeds diagnostic resource bound");
    NSData *q=fixtureData(base,in.manifest[@"query_file"],size_t(in.hq)*in.d*4);
    in.q.assign((const float *)q.bytes,(const float *)q.bytes+q.length/4);
    NSMutableDictionary *prov=[@{@"manifest_path":absolute,@"manifest_sha256":hashBytes(raw.bytes,raw.length),
        @"query_file":in.manifest[@"query_file"],@"query_sha256":hashBytes(q.bytes,q.length),
        @"query_precision":@"unchanged FP32 fixture Q for every variant"} mutableCopy];
    auto loadSide=[&](NSString *side) {
        NSString *overrideKey=[NSString stringWithFormat:@"baseline_%@_f32_file",side];
        NSString *storage=in.manifest[[side stringByAppendingString:@"_storage"]];
        bool half=[storage isEqual:@"fp16"] && !in.manifest[overrideKey];
        NSString *key=in.manifest[overrideKey]?overrideKey:[side stringByAppendingString:half?@"_file":@"_decoded_file"];
        id file=in.manifest[key];
        NSData *rawSide=fixtureData(base,file,count*(half?2:4));
        std::vector<float> values(count);
        if(half) ggml_fp16_to_fp32_row((const ggml_fp16_t *)rawSide.bytes,values.data(),count);
        else std::memcpy(values.data(),rawSide.bytes,count*4);
        checkFinite(values);
        // Both F16 and Q4_0 consume this same F16 base; no extra rounding of Q.
        std::vector<ggml_fp16_t> f16(count);
        ggml_fp32_to_fp16_row(values.data(),f16.data(),count);
        ggml_fp16_to_fp32_row(f16.data(),values.data(),count);checkFinite(values);
        prov[[side stringByAppendingString:@"_file"]]=file;
        prov[[side stringByAppendingString:@"_file_sha256"]]=hashBytes(rawSide.bytes,rawSide.length);
        prov[[side stringByAppendingString:@"_source_kind"]]=half?@"original fixture F16":(in.manifest[overrideKey]?@"explicit baseline F32 override rounded to F16":@"offline decoded fixture F32 rounded to F16");
        prov[[side stringByAppendingString:@"_base_f16_sha256"]]=hashBytes(f16.data(),f16.size()*2);
        return values;
    };
    in.k=loadSide(@"k");in.v=loadSide(@"v");checkFinite(in.q);in.provenance=prov;return in;
}
static std::vector<float> reference(const Input &in,const std::vector<float> &k,const std::vector<float> &v) {
    std::vector<float> out(size_t(in.hq)*in.d);std::vector<double> scores(in.t);
    for(int64_t h=0;h<in.hq;++h) {
        int64_t kh=h/(in.hq/in.hkv);double top=-INFINITY;
        for(int64_t t=0;t<in.t;++t) {
            double dot=0;for(int64_t c=0;c<in.d;++c) dot+=double(in.q[h*in.d+c])*k[(kh*in.t+t)*in.d+c];
            scores[t]=dot/std::sqrt(double(in.d));top=std::max(top,scores[t]);
        }
        double denom=0;for(double &s:scores) {s=std::exp(s-top);denom+=s;}
        for(int64_t c=0;c<in.d;++c) {
            double sum=0;for(int64_t t=0;t<in.t;++t) sum+=scores[t]*v[(kh*in.t+t)*in.d+c];
            out[h*in.d+c]=float(sum/denom);
        }
    }
    checkFinite(out);return out;
}
static NSDictionary *errorStats(const std::vector<float> &actual,const std::vector<float> &expected) {
    require(actual.size()==expected.size(),"Reference shape mismatch");
    double squared=0,norm=0,maxabs=0,refmax=0;
    for(size_t i=0;i<actual.size();++i) {
        if(!std::isfinite(actual[i])||!std::isfinite(expected[i])) return @{@"finite":@NO,@"passed":@NO};
        double e=double(actual[i])-expected[i];squared+=e*e;norm+=double(expected[i])*expected[i];
        maxabs=std::max(maxabs,std::abs(e));refmax=std::max(refmax,std::abs(double(expected[i])));
    }
    double relative=std::sqrt(squared/std::max(norm,1e-30));
    bool pass=std::isfinite(relative)&&relative<=1e-3&&maxabs<=1e-3*std::max(1.0,refmax);
    return @{@"finite":@YES,@"relative_l2":@(relative),@"max_abs":@(maxabs),@"reference_max_abs":@(refmax),@"passed":@(pass),
        @"criterion":@"relative_L2 <= 1e-3 AND max_abs <= 1e-3 * max(1,reference_max_abs)"};
}
struct Encoded { std::vector<uint8_t> bytes;std::vector<float> decoded; };
static Encoded encode(const std::vector<float> &source,ggml_type type,int64_t d) {
    Encoded e;size_t rows=source.size()/d,rowbytes=ggml_row_size(type,d);
    e.bytes.resize(rows*rowbytes);e.decoded.resize(source.size());
    size_t written=ggml_quantize_chunk(type,source.data(),e.bytes.data(),0,rows,d,nullptr);
    require(written==e.bytes.size(),"Official quantizer returned incorrect size");
    const auto *traits=ggml_get_type_traits(type);require(traits&&traits->to_float,"Missing official decoder");
    for(size_t r=0;r<rows;++r) traits->to_float(e.bytes.data()+r*rowbytes,e.decoded.data()+r*d,d);
    checkFinite(e.decoded);return e;
}
static std::mutex logMutex;
static std::vector<std::string> pipelineEvents;
static std::atomic<bool> timed{false};
static std::atomic<unsigned> newPipelinesDuringTiming{0};
static void logCallback(ggml_log_level level,const char *message,void *) {
    std::string s=message?message:"";
    bool pipeline=s.find("compiling pipeline:")!=std::string::npos&&s.find("flash_attn")!=std::string::npos;
    if(pipeline) {
        std::lock_guard<std::mutex> guard(logMutex);
        if(pipelineEvents.size()<256) pipelineEvents.push_back(s.substr(0,2048));
        if(timed) ++newPipelinesDuringTiming;
    }
    if(!timed || level>=GGML_LOG_LEVEL_WARN) std::fputs(s.c_str(),stderr);
}
static NSArray *eventsSince(size_t start) {
    std::lock_guard<std::mutex> guard(logMutex);NSMutableArray *a=[NSMutableArray new];
    for(size_t i=start;i<pipelineEvents.size();++i) [a addObject:ns(pipelineEvents[i].c_str())];return a;
}
struct Case {
    std::string name;ggml_type kt,vt;ggml_context *ctx=nullptr;ggml_cgraph *graph=nullptr;
    ggml_tensor *q=nullptr,*k=nullptr,*v=nullptr,*out=nullptr;ggml_backend_buffer_t buffer=nullptr;
    Encoded ke,ve;std::vector<float> ref,output;bool supported=false;NSDictionary *validation=nil;NSArray *pipelines=nil;
    ~Case(){if(buffer)ggml_backend_buffer_free(buffer);if(ctx)ggml_free(ctx);}
};
static std::unique_ptr<Case> makeCase(const Input &in,ggml_backend_t backend,const char *name,ggml_type kt,ggml_type vt) {
    auto c=std::make_unique<Case>();c->name=name;c->kt=kt;c->vt=vt;
    ggml_init_params init{32*ggml_tensor_overhead()+ggml_graph_overhead_custom(16,false),nullptr,true};
    c->ctx=ggml_init(init);require(c->ctx,"Cannot create ggml context");
    c->q=ggml_new_tensor_4d(c->ctx,GGML_TYPE_F32,in.d,1,in.hq,1);
    c->k=ggml_new_tensor_4d(c->ctx,kt,in.d,in.t,in.hkv,1);
    c->v=ggml_new_tensor_4d(c->ctx,vt,in.d,in.t,in.hkv,1);
    c->out=ggml_flash_attn_ext(c->ctx,c->q,c->k,c->v,nullptr,1/std::sqrt(float(in.d)),0,0);
    require(ggml_prec_set_acc(c->out,GGML_PREC_F32),"Cannot request FP32 accumulation");
    ggml_set_name(c->q,"same_fp32_query");ggml_set_name(c->k,"represented_k");ggml_set_name(c->v,"represented_v");ggml_set_name(c->out,"fused_attention");
    c->graph=ggml_new_graph_custom(c->ctx,16,false);ggml_build_forward_expand(c->graph,c->out);
    require(ggml_graph_n_nodes(c->graph)==1&&ggml_graph_node(c->graph,0)->op==GGML_OP_FLASH_ATTN_EXT,"Graph must contain exactly one fused attention op");
    c->supported=ggml_backend_supports_op(backend,c->out);if(!c->supported)return c;
    c->buffer=ggml_backend_alloc_ctx_tensors(c->ctx,backend);require(c->buffer,"Cannot allocate backend tensors/scratch");
    c->ke=encode(in.k,kt,in.d);c->ve=encode(in.v,vt,in.d);
    require(ggml_nbytes(c->k)==c->ke.bytes.size()&&ggml_nbytes(c->v)==c->ve.bytes.size(),"Packed row layout mismatch");
    ggml_backend_tensor_set(c->q,in.q.data(),0,in.q.size()*4);
    ggml_backend_tensor_set(c->k,c->ke.bytes.data(),0,c->ke.bytes.size());ggml_backend_tensor_set(c->v,c->ve.bytes.data(),0,c->ve.bytes.size());
    c->ref=reference(in,c->ke.decoded,c->ve.decoded);c->output.resize(in.q.size());return c;
}
static double run(Case &c,ggml_backend_t backend) {
    auto start=Clock::now();
    auto status=ggml_backend_graph_compute(backend,c.graph);ggml_backend_synchronize(backend);
    double elapsed=ms(start);require(status==GGML_STATUS_SUCCESS,"ggml backend graph compute failed");return elapsed;
}
static Input syntheticInput() {
    Input in;in.t=96;in.hq=9;in.hkv=3;in.d=64;in.q.assign(in.hq*in.d,0);in.k.resize(in.hkv*in.t*in.d);in.v.resize(in.k.size());
    for(int64_t h=0;h<in.hkv;++h)for(int64_t t=0;t<in.t;++t)for(int64_t c=0;c<in.d;++c) {
        // Multiples of 0.5 keep the official CPU F16 accumulator exact over T96.
        // This is a glue/GQA test, not a CPU accuracy certificate for real inputs.
        size_t i=(h*in.t+t)*in.d+c;in.k[i]=float((t+c)%31-15)/16;in.v[i]=float(h+1)+float(c%2)/2;
    }
    in.manifest=@{@"kind":@"synthetic zero-Q GQA and quantized-row contract check"};in.provenance=@{};return in;
}
int main(int argc,const char **argv) {
    @autoreleasepool {try {
        require(argc>=2,"Usage: astc-llama-baseline MANIFEST [rounds=30] [seed=20261028] | --cpu-self-test");
        bool cpu=std::string(argv[1])=="--cpu-self-test";
        unsigned rounds=argc>2?unsigned(std::stoul(argv[2])):30,seed=argc>3?unsigned(std::stoul(argv[3])):20261028;
        require(rounds>=1&&rounds<=10000,"rounds must be in [1,10000]");
        ggml_log_set(logCallback,nullptr);ggml_time_init();
        Input in=cpu?syntheticInput():loadInput(ns(argv[1]));
        auto setup=Clock::now();ggml_backend_t backend=cpu?ggml_backend_cpu_init():ggml_backend_metal_init();
        require(backend!=nullptr,"Requested backend unavailable; no fallback executed");
        std::unique_ptr<ggml_backend,std::function<void(ggml_backend_t)>> owner(backend,ggml_backend_free);
        if(cpu)ggml_backend_cpu_set_n_threads(backend,1);else require(ggml_backend_is_metal(backend),"Expected official Metal backend");
        ggml_backend_dev_props props{};ggml_backend_dev_get_props(ggml_backend_get_device(backend),&props);
        std::vector<std::unique_ptr<Case>> cases;
        cases.push_back(makeCase(in,backend,"f16_kv",GGML_TYPE_F16,GGML_TYPE_F16));
        cases.push_back(makeCase(in,backend,"q4_0_kv",GGML_TYPE_Q4_0,GGML_TYPE_Q4_0));
        if(!cpu) {
            cases.push_back(makeCase(in,backend,"q4_0_k_f16_v",GGML_TYPE_Q4_0,GGML_TYPE_F16));
            cases.push_back(makeCase(in,backend,"f16_k_q4_0_v",GGML_TYPE_F16,GGML_TYPE_Q4_0));
        }
        bool pass=true;std::vector<int> active;NSMutableDictionary *variants=[NSMutableDictionary new];
        for(size_t i=0;i<cases.size();++i) {
            Case &c=*cases[i];NSString *name=ns(c.name.c_str());
            if(!c.supported) {variants[name]=@{@"status":@"unsupported_by_selected_backend",@"fallback_executed":@NO};continue;}
            active.push_back(int(i));size_t eventStart=pipelineEvents.size();run(c,backend);
            std::vector<float> qAfter(in.q.size());ggml_backend_tensor_get(c.q,qAfter.data(),0,qAfter.size()*4);
            require(std::memcmp(qAfter.data(),in.q.data(),qAfter.size()*4)==0,"Backend modified the shared query values");
            ggml_backend_tensor_get(c.out,c.output.data(),0,c.output.size()*4);c.validation=errorStats(c.output,c.ref);c.pipelines=eventsSince(eventStart);
            pass=pass&&[c.validation[@"passed"] boolValue];
            if(cpu) {
                // Zero Q gives uniform probabilities: independent scalar GQA mean.
                std::vector<float> oracle(in.q.size());
                for(int64_t h=0;h<in.hq;++h)for(int64_t d=0;d<in.d;++d) {
                    double sum=0;for(int64_t t=0;t<in.t;++t)sum+=c.ve.decoded[((h/(in.hq/in.hkv))*in.t+t)*in.d+d];
                    oracle[h*in.d+d]=float(sum/in.t);
                }
                pass=pass&&[errorStats(c.output,oracle)[@"passed"] boolValue];
            }
            variants[name]=@{@"status":@"executed",@"k_type":ns(ggml_type_name(c.kt)),@"v_type":ns(ggml_type_name(c.vt)),
                @"same_values_fp64_reference":c.validation,@"packed_k_sha256":hashBytes(c.ke.bytes.data(),c.ke.bytes.size()),
                @"packed_v_sha256":hashBytes(c.ve.bytes.data(),c.ve.bytes.size()),@"logical_kv_bytes":@(c.ke.bytes.size()+c.ve.bytes.size()),
                @"effective_kv_bits":@(8.0*(c.ke.bytes.size()+c.ve.bytes.size())/(2*in.k.size())),
                @"backend_buffer_bytes_with_graph_scratch":@(ggml_backend_buffer_get_size(c.buffer)),
                @"graph_nodes":@(ggml_graph_n_nodes(c.graph)),@"graph_op":@"GGML_OP_FLASH_ATTN_EXT",@"pipeline_messages":c.pipelines,
                @"query_unchanged_after_correctness_call":@YES,
                @"output_fp32_sha256":hashBytes(c.output.data(),c.output.size()*4)};
        }
        require(!active.empty(),"No supported fused variant");
        NSMutableDictionary *result=[@{@"schema":@"astc-kv-llama-fused-baseline-v1",@"status":pass?(cpu?@"cpu_self_test_passed":@"correctness_passed"):@"correctness_failed",
            @"source_revision":@AKV_LLAMA_REVISION,@"input_manifest":in.manifest,@"input_provenance":in.provenance,
            @"backend":ns(ggml_backend_name(backend)),@"device":ns(props.description),@"physical_memory_bytes":@(props.memory_total),
            @"shape":@{@"D":@(in.d),@"T":@(in.t),@"Hq":@(in.hq),@"Hkv":@(in.hkv),@"query_tokens":@1},
            @"variants":variants,@"setup_wall_ms":@(ms(setup)),@"raw_samples":@[],
            @"scope":@"Standalone fused attention only; no model latency or quality claim"} mutableCopy];
        NSData *buildInfo=[NSData dataWithContentsOfFile:@AKV_BUILD_INFO_PATH];
        if(buildInfo)result[@"build_provenance"]=[NSJSONSerialization JSONObjectWithData:buildInfo options:0 error:nil]?:[NSNull null];
        if(!pass||cpu) {printJSON(result);return pass?0:2;}
        std::mt19937 rng(seed);for(unsigned w=0;w<10;++w) {std::shuffle(active.begin(),active.end(),rng);for(int i:active)run(*cases[i],backend);}
        NSMutableArray *samples=[NSMutableArray new];timed=true;
        for(unsigned r=0;r<rounds;++r) {
            std::shuffle(active.begin(),active.end(),rng);
            for(size_t pos=0;pos<active.size();++pos) {Case &c=*cases[active[pos]];double elapsed=run(c,backend);
                [samples addObject:@{@"variant":ns(c.name.c_str()),@"round":@(r),@"order_within_round":@(pos),@"host_sync_ms":@(elapsed)}];}
        }
        timed=false;result[@"raw_samples"]=samples;result[@"status"]=@"executed_on_metal";
        result[@"timing"]=@{@"rounds":@(rounds),@"warmup_calls_per_variant":@10,@"order_seed":@(seed),
            @"boundary":@"one ggml_backend_graph_compute + ggml_backend_synchronize, monotonic host clock; output readback excluded",
            @"new_pipelines_during_timing":@(newPipelinesDuringTiming.load()),@"gpu_time_ms":[NSNull null],
            @"cache_state":@"repeated resident one-query fixture; not a bandwidth or whole-model benchmark"};
        result[@"timing_valid"]=@(newPipelinesDuringTiming==0);printJSON(result);return newPipelinesDuringTiming==0?0:3;
    }catch(const std::exception &e){timed=false;std::fprintf(stderr,"astc-llama-baseline: %s\n",e.what());return 1;}}
}
