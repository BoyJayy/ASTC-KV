// Separate diagnosis: original benchmark source and its acceptance gate stay unchanged.
#define AKV_LLAMA_REVISION "c3df4b7e22a1ac38da663552d19e649dd97ca936"
#define AKV_BUILD_INFO_PATH "build/llama-baseline/build-provenance.json"
#define main original_baseline_main
#include "probe.mm"
#undef main

static std::vector<float> halfRound(const std::vector<float> &values) {
    std::vector<ggml_fp16_t> half(values.size());std::vector<float> out(values.size());
    ggml_fp32_to_fp16_row(values.data(),half.data(),values.size());
    ggml_fp16_to_fp32_row(half.data(),out.data(),out.size());checkFinite(out);return out;
}
static NSString *saveFloat(NSString *dir,NSString *name,const std::vector<float> &values) {
    NSString *path=[dir stringByAppendingPathComponent:name];
    require(![[NSFileManager defaultManager] fileExistsAtPath:path],"Refusing to overwrite precision diagnosis output");
    NSData *data=[NSData dataWithBytes:values.data() length:values.size()*4];
    require([data writeToFile:path atomically:YES],"Cannot save precision diagnosis output");
    return hashBytes(values.data(),values.size()*4);
}
static NSDictionary *deltaAlignment(const std::vector<float> &actual,const std::vector<float> &base,const std::vector<float> &predicted) {
    double ab=0,aa=0,bb=0,residual=0;
    for(size_t i=0;i<base.size();++i) {double a=double(actual[i])-base[i],b=double(predicted[i])-base[i];
        ab+=a*b;aa+=a*a;bb+=b*b;residual+=(a-b)*(a-b);}
    return @{@"cosine_of_actual_and_predicted_deltas":@(ab/std::sqrt(std::max(aa*bb,1e-60))),
        @"remaining_error_l2_fraction":@(std::sqrt(residual/std::max(aa,1e-30)))};
}
int main(int argc,const char **argv) {
    @autoreleasepool {try {
        bool cpu=argc>1 && std::string(argv[1])=="--cpu-prediction";
        int shift=cpu?1:0;
        require(argc==3+shift,"Usage: astc-llama-precision [--cpu-prediction] MANIFEST NEW_OUTPUT_DIR");
        ggml_log_set(logCallback,nullptr);ggml_time_init();
        Input in=loadInput(ns(argv[1+shift]));Input halfQ=in;halfQ.q=halfRound(in.q);
        NSString *dir=ns(argv[2+shift]);BOOL isDir=NO;
        if([[NSFileManager defaultManager] fileExistsAtPath:dir isDirectory:&isDir]) require(isDir,"Output path is not a directory");
        else require([[NSFileManager defaultManager] createDirectoryAtPath:dir withIntermediateDirectories:YES attributes:nil error:nil],"Cannot create output directory");
        ggml_backend_t backend=cpu?nullptr:ggml_backend_metal_init();
        if(!cpu)require(backend&&ggml_backend_is_metal(backend),"Metal unavailable; no fallback");
        std::unique_ptr<ggml_backend,std::function<void(ggml_backend_t)>> owner(backend,[](ggml_backend_t p){if(p)ggml_backend_free(p);});
        NSMutableDictionary *variants=[NSMutableDictionary new];
        for(auto type:{GGML_TYPE_F16,GGML_TYPE_Q4_0}) {
            NSString *name=type==GGML_TYPE_F16?@"f16_kv":@"q4_0_kv";
            std::unique_ptr<Case> c;
            if(cpu) {c=std::make_unique<Case>();c->ke=encode(in.k,type,in.d);c->ve=encode(in.v,type,in.d);c->ref=reference(in,c->ke.decoded,c->ve.decoded);}
            else {c=makeCase(in,backend,name.UTF8String,type,type);require(c->supported,"Requested fused variant unsupported");}
            auto refA=c->ref;
            auto refB=reference(halfQ,c->ke.decoded,c->ve.decoded);
            auto refC=reference(halfQ,halfRound(c->ke.decoded),halfRound(c->ve.decoded));
            NSMutableDictionary *files=[NSMutableDictionary new];
            files[@"reference_A_original_q_cpu_decode"] = saveFloat(dir,[name stringByAppendingString:@"_reference_A.f32"],refA);
            files[@"reference_B_half_q_cpu_decode"] = saveFloat(dir,[name stringByAppendingString:@"_reference_B.f32"],refB);
            files[@"reference_C_half_q_half_decode"] = saveFloat(dir,[name stringByAppendingString:@"_reference_C.f32"],refC);
            NSMutableDictionary *r=[@{@"B_vs_A_predicted_query_rounding_effect":errorStats(refB,refA),
                @"C_vs_B_predicted_dequant_rounding_effect":errorStats(refC,refB),
                @"C_vs_A_predicted_combined_effect":errorStats(refC,refA),
                @"file_sha256":files,@"packed_k_sha256":hashBytes(c->ke.bytes.data(),c->ke.bytes.size()),
                @"packed_v_sha256":hashBytes(c->ve.bytes.data(),c->ve.bytes.size())} mutableCopy];
            if(!cpu) {
                size_t events=pipelineEvents.size();run(*c,backend);ggml_backend_tensor_get(c->out,c->output.data(),0,c->output.size()*4);
                files[@"actual_gpu_output"] = saveFloat(dir,[name stringByAppendingString:@"_actual.f32"],c->output);
                r[@"actual_vs_A_original_reference"]=errorStats(c->output,refA);
                r[@"actual_vs_B_half_query_reference"]=errorStats(c->output,refB);
                r[@"actual_vs_C_half_query_half_decode_reference"]=errorStats(c->output,refC);
                r[@"query_rounding_explains_delta"]=deltaAlignment(c->output,refA,refB);
                r[@"combined_rounding_explains_delta"]=deltaAlignment(c->output,refA,refC);
                r[@"pipeline_messages"]=eventsSince(events);
            }
            variants[name]=r;
        }
        printJSON(@{@"schema":@"llama-fused-precision-diagnosis-v1",@"status":cpu?@"cpu_predictions_only":@"gpu_diagnosis_executed",
            @"source_revision":@AKV_LLAMA_REVISION,@"input_provenance":in.provenance,@"variants":variants,
            @"scope":@"Separate precision attribution; no benchmark, no threshold change, no timing result",
            @"hypothesis":@"Official FA_TYPES use half4 Q/K/V, float QK/softmax/output; references B and C reflect explicit operand rounding",
            @"caveat":@"Reference C rounds official CPU dequant to half; extreme/subnormal Q4 scale arithmetic may need a more detailed decoder model"});
        return 0;
    }catch(const std::exception &e){std::fprintf(stderr,"llama precision diagnosis: %s\n",e.what());return 1;}}
}
