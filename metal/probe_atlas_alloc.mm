#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>

// Allocation only: no bytes are uploaded and no GPU command is submitted.
static void require(bool ok,const char *message) {
    if(!ok) throw std::runtime_error(message);
}
static uint64_t ceilDiv(uint64_t a,uint64_t b) { return (a+b-1)/b; }

int main(int argc,const char **) {
    @autoreleasepool {
        try {
            require(argc==1,"Usage: astc-kv-atlas-alloc (fixed bounded cases, no arguments)");
            id<MTLDevice> device=MTLCreateSystemDefaultDevice();
            require(device!=nil,"Metal device is unavailable; allocation probe did not run");
            constexpr uint64_t maxDimension=16384,liveBudget=64ull*1024*1024;
            constexpr uint64_t blockWidth=6,blockHeight=6,blockBytes=16,heads=3,page=96;
            const uint64_t baselineAllocated=device.currentAllocatedSize;
            NSMutableArray *cases=[NSMutableArray new];
            bool allEligibleSucceeded=true;
            unsigned allocatedCount=0;
            for(uint64_t tokens:{384ull,2048ull,8192ull}) {
                const uint64_t paddedHeight=ceilDiv(tokens,page)*page;
                const uint64_t originalBlocks=heads*11*(paddedHeight/blockHeight);
                const uint64_t originalPayload=originalBlocks*blockBytes;
                auto measure=[&](NSString *layout,uint64_t widthBlocks,uint64_t heightBlocks,
                                 uint64_t slices) {
                    const uint64_t width=widthBlocks*blockWidth,height=heightBlocks*blockHeight;
                    const uint64_t capacityBlocks=widthBlocks*heightBlocks*slices;
                    require(capacityBlocks>=originalBlocks,"Layout has insufficient original block capacity");
                    NSMutableDictionary *row=[@{
                        @"tokens":@(tokens),@"padded_token_height":@(paddedHeight),@"layout":layout,
                        @"width":@(width),@"height":@(height),@"array_length":@(slices),
                        @"width_blocks":@(widthBlocks),@"height_blocks":@(heightBlocks),
                        @"original_block_count":@(originalBlocks),@"capacity_blocks":@(capacityBlocks),
                        @"original_payload_bytes":@(originalPayload),@"capacity_payload_bytes":@(capacityBlocks*blockBytes),
                        @"additional_padding_blocks":@(capacityBlocks-originalBlocks),
                        @"additional_padding_bytes":@((capacityBlocks-originalBlocks)*blockBytes),
                        @"payload_padding_ratio":@(double(capacityBlocks)/originalBlocks),
                        @"has_original_block_capacity":@YES,@"exact_original_block_capacity":@(capacityBlocks==originalBlocks),
                        @"texture_allocated_bytes":[NSNull null],@"allocation_succeeded":@NO
                    } mutableCopy];
                    if(width>maxDimension || height>maxDimension) {
                        row[@"status"]=@"skipped_dimension_bound";
                    } else if(capacityBlocks*5>originalBlocks*6) {
                        row[@"status"]=@"skipped_padding_over_1_2";
                    } else {
                        // Scope keeps at most one texture live; rows retain only scalar observations.
                        @autoreleasepool {
                            auto desc=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatASTC_6x6_LDR
                                                                             width:width height:height mipmapped:NO];
                            desc.textureType=MTLTextureType2DArray;
                            desc.arrayLength=slices;
                            desc.storageMode=MTLStorageModePrivate;
                            desc.usage=MTLTextureUsageShaderRead;
                            const MTLSizeAndAlign estimate=[device heapTextureSizeAndAlignWithDescriptor:desc];
                            const uint64_t before=device.currentAllocatedSize;
                            const uint64_t ownLive=before>baselineAllocated?before-baselineAllocated:0;
                            row[@"heap_texture_size_estimate_bytes"]=@(estimate.size);
                            row[@"device_current_allocated_before_bytes"]=@(before);
                            if(ownLive>liveBudget || estimate.size>liveBudget-ownLive) {
                                row[@"status"]=@"skipped_live_budget";
                                allEligibleSucceeded=false;
                            } else {
                                id<MTLTexture> texture=[device newTextureWithDescriptor:desc];
                                if(!texture) {
                                    row[@"status"]=@"rejected_by_device";
                                    allEligibleSucceeded=false;
                                } else {
                                    texture.label=layout;
                                    const uint64_t allocated=texture.allocatedSize;
                                    require(allocated<=liveBudget-ownLive,"Actual allocation exceeds 64 MiB live budget");
                                    row[@"status"]=@"allocated";
                                    row[@"allocation_succeeded"]=@YES;
                                    row[@"texture_allocated_bytes"]=@(allocated);
                                    row[@"allocation_to_capacity_payload_ratio"]=@(double(allocated)/(capacityBlocks*blockBytes));
                                    row[@"allocation_to_original_payload_ratio"]=@(double(allocated)/originalPayload);
                                    row[@"device_current_allocated_with_texture_bytes"]=@(device.currentAllocatedSize);
                                    ++allocatedCount;
                                }
                            }
                        }
                        row[@"device_current_allocated_after_release_bytes"]=@(device.currentAllocatedSize);
                    }
                    [cases addObject:row];
                };
                measure(@"original_3_slice_66_wide",11,paddedHeight/blockHeight,heads);
                measure(@"head_horizontal_198_wide",33,paddedHeight/blockHeight,1);
                for(uint64_t widthBlocks:{32ull,64ull,128ull,256ull})
                    measure([NSString stringWithFormat:@"compact_%llu_blocks_wide",(unsigned long long)widthBlocks],
                            widthBlocks,ceilDiv(originalBlocks,widthBlocks),1);
            }
            NSDictionary *result=@{
                @"schema":@"astc-kv-atlas-allocation-v1",@"status":@"allocation_only_executed",
                @"data_uploaded":@NO,@"gpu_commands_submitted":@NO,@"decode_validated":@NO,
                @"attention_validated":@NO,@"codec_quality_validated":@NO,@"performance_measured":@NO,
                @"device":@{@"name":device.name,@"registry_id":@(device.registryID),
                    @"supports_apple7":@([device supportsFamily:MTLGPUFamilyApple7]),
                    @"unified_memory":@(device.hasUnifiedMemory),
                    @"recommended_max_working_set_bytes":@(device.recommendedMaxWorkingSetSize),
                    @"os":[NSProcessInfo processInfo].operatingSystemVersionString},
                @"format":@"ASTC_6x6_LDR",@"texture_type":@"2DArray",@"storage_mode":@"private",
                @"usage":@"shader_read",@"max_dimension":@(maxDimension),
                @"max_live_probe_allocated_bytes":@(liveBudget),@"max_payload_padding_ratio":@1.2,
                @"baseline_device_current_allocated_bytes":@(baselineAllocated),
                @"final_device_current_allocated_bytes":@(device.currentAllocatedSize),
                @"all_eligible_allocations_succeeded":@(allEligibleSucceeded),@"allocated_case_count":@(allocatedCount),
                @"cases":cases,
                @"scope":@"Allocation sizes only. Block capacity does not validate block remapping, coordinate addressing, decoder output, codec quality, physical residency or attention speed. Existing current-layout negative performance result is unchanged."
            };
            NSError *error=nil;
            NSData *json=[NSJSONSerialization dataWithJSONObject:result
                       options:NSJSONWritingPrettyPrinted|NSJSONWritingSortedKeys error:&error];
            require(json!=nil,"Cannot serialize result");
            std::fwrite(json.bytes,1,json.length,stdout); std::fputc('\n',stdout);
            return allEligibleSucceeded?0:2;
        } catch(const std::exception &e) {
            std::fprintf(stderr,"astc-kv-atlas-alloc: %s\n",e.what());
            return 1;
        }
    }
}
