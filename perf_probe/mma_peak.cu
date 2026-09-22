// Pure tensor-core throughput microbenchmark for sm120 (consumer Blackwell).
// Hammers mma.sync in a register-resident loop (no memory traffic) to isolate
// the tensor-core issue rate and expose any FP32-accumulate throttle.
//
// Paths measured:
//   1. FP16 in / FP16 acc   m16n8k16   (full-rate baseline)
//   2. FP16 in / FP32 acc   m16n8k16   (what bf16 attention actually uses)
//   3. FP8 e4m3 / FP32 acc  m16n8k32   (target path)
#include <cstdio>
#include <cuda_runtime.h>

#ifndef NACC
#define NACC 8          // independent accumulator chains per thread (ILP to hide MMA latency)
#endif
#ifndef ITERS
#define ITERS 120000    // inner loop count
#endif
#define BLOCK 256       // threads per block (8 warps)

#define CK(x) do{ cudaError_t e=(x); if(e){printf("CUDA err %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(e)); exit(1);} }while(0)

// ---- Kernel 1: FP16 inputs, FP16 accumulate, m16n8k16 ----
__global__ void k_f16f16(float* out){
    unsigned a0=0x3C003C00,a1=0x3C003C00,a2=0x3C003C00,a3=0x3C003C00;
    unsigned b0=0x3C003C00,b1=0x3C003C00;
    unsigned c[NACC][2];
    #pragma unroll
    for(int j=0;j<NACC;j++){ c[j][0]=0; c[j][1]=0; }
    for(int it=0; it<ITERS; ++it){
        #pragma unroll
        for(int j=0;j<NACC;j++){
            asm volatile(
              "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
              "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
              : "+r"(c[j][0]),"+r"(c[j][1])
              : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
        }
    }
    unsigned acc=0;
    #pragma unroll
    for(int j=0;j<NACC;j++) acc ^= c[j][0]^c[j][1];
    if(acc==0xFFFFFFFF) out[threadIdx.x]= (float)acc; // anti-DCE
}

// ---- Kernel 2: FP16 inputs, FP32 accumulate, m16n8k16 ----
__global__ void k_f16f32(float* out){
    unsigned a0=0x3C003C00,a1=0x3C003C00,a2=0x3C003C00,a3=0x3C003C00;
    unsigned b0=0x3C003C00,b1=0x3C003C00;
    float c[NACC][4];
    #pragma unroll
    for(int j=0;j<NACC;j++){ c[j][0]=c[j][1]=c[j][2]=c[j][3]=0.f; }
    for(int it=0; it<ITERS; ++it){
        #pragma unroll
        for(int j=0;j<NACC;j++){
            asm volatile(
              "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
              "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
              : "+f"(c[j][0]),"+f"(c[j][1]),"+f"(c[j][2]),"+f"(c[j][3])
              : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
        }
    }
    float acc=0;
    #pragma unroll
    for(int j=0;j<NACC;j++) acc += c[j][0]+c[j][1]+c[j][2]+c[j][3];
    if(acc==-1.f) out[threadIdx.x]=acc; // anti-DCE
}

// ---- Kernel 3: FP8 e4m3 inputs, FP32 accumulate, m16n8k32 ----
__global__ void k_e4m3f32(float* out){
    // A: 16 e4m3/thread = 4 regs; B: 8 e4m3/thread = 2 regs
    unsigned a0=0x38383838,a1=0x38383838,a2=0x38383838,a3=0x38383838; // e4m3 ~1.0 bytes
    unsigned b0=0x38383838,b1=0x38383838;
    float c[NACC][4];
    #pragma unroll
    for(int j=0;j<NACC;j++){ c[j][0]=c[j][1]=c[j][2]=c[j][3]=0.f; }
    for(int it=0; it<ITERS; ++it){
        #pragma unroll
        for(int j=0;j<NACC;j++){
            asm volatile(
              "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
              "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
              : "+f"(c[j][0]),"+f"(c[j][1]),"+f"(c[j][2]),"+f"(c[j][3])
              : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
        }
    }
    float acc=0;
    #pragma unroll
    for(int j=0;j<NACC;j++) acc += c[j][0]+c[j][1]+c[j][2]+c[j][3];
    if(acc==-1.f) out[threadIdx.x]=acc; // anti-DCE
}

// ---- Kernel 4: FP8 e4m3 inputs, FP16 accumulate, m16n8k32 ----
__global__ void k_e4m3f16(float* out){
    unsigned a0=0x38383838,a1=0x38383838,a2=0x38383838,a3=0x38383838;
    unsigned b0=0x38383838,b1=0x38383838;
    unsigned c[NACC][2]; // f16 accumulate -> 2 regs
    #pragma unroll
    for(int j=0;j<NACC;j++){ c[j][0]=0; c[j][1]=0; }
    for(int it=0; it<ITERS; ++it){
        #pragma unroll
        for(int j=0;j<NACC;j++){
            asm volatile(
              "mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 "
              "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
              : "+r"(c[j][0]),"+r"(c[j][1])
              : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
        }
    }
    unsigned acc=0;
    #pragma unroll
    for(int j=0;j<NACC;j++) acc ^= c[j][0]^c[j][1];
    if(acc==0xFFFFFFFF) out[threadIdx.x]=(float)acc; // anti-DCE
}

// ---- Kernel 5: MXFP8 block-scaled (e4m3 + ue8m0), FP32 accumulate, m16n8k32 ----
__global__ void k_mxf8(float* out){
    unsigned a0=0x38383838,a1=0x38383838,a2=0x38383838,a3=0x38383838;
    unsigned b0=0x38383838,b1=0x38383838;
    unsigned sfa=127u, sfb=127u; unsigned short bid=0,tid=0;
    float c[NACC][4];
    #pragma unroll
    for(int j=0;j<NACC;j++){ c[j][0]=c[j][1]=c[j][2]=c[j][3]=0.f; }
    for(int it=0; it<ITERS; ++it){
        #pragma unroll
        for(int j=0;j<NACC;j++){
            asm volatile(
              "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
              "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, "
              "{%10}, {%11,%12}, {%13}, {%14,%15};\n"
              : "+f"(c[j][0]),"+f"(c[j][1]),"+f"(c[j][2]),"+f"(c[j][3])
              : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),
                "r"(sfa),"h"(bid),"h"(tid),"r"(sfb),"h"(bid),"h"(tid));
        }
    }
    float acc=0;
    #pragma unroll
    for(int j=0;j<NACC;j++) acc += c[j][0]+c[j][1]+c[j][2]+c[j][3];
    if(acc==-1.f) out[threadIdx.x]=acc;
}

template<class F>
double run(F launch, int gridBlocks, double flopsPerWarpMMA){
    float* out; CK(cudaMalloc(&out, BLOCK*sizeof(float)));
    // warmup
    launch(out); CK(cudaDeviceSynchronize());
    cudaEvent_t evS,evE; CK(cudaEventCreate(&evS)); CK(cudaEventCreate(&evE));
    const int RUNS=3; float best=1e30f;
    for(int r=0;r<RUNS;r++){
        CK(cudaEventRecord(evS));
        launch(out);
        CK(cudaEventRecord(evE)); CK(cudaEventSynchronize(evE));
        float ms; CK(cudaEventElapsedTime(&ms,evS,evE));
        if(ms<best) best=ms;
    }
    double warps = (double)gridBlocks * (BLOCK/32);
    double warpMMAs = warps * NACC * (double)ITERS;
    double tflops = warpMMAs * flopsPerWarpMMA / (best/1e3) / 1e12;
    cudaFree(out);
    return tflops;
}

int main(){
    int dev=0; cudaDeviceProp p; CK(cudaGetDeviceProperties(&p,dev));
    int sm=p.multiProcessorCount;
    int gridBlocks = sm * 4; // 4 blocks/SM
    printf("GPU: %s  SMs=%d  sm_%d%d  grid=%d blocks x %d thr  NACC=%d ITERS=%d\n",
           p.name, sm, p.major, p.minor, gridBlocks, BLOCK, NACC, ITERS);
    printf("%-26s %12s\n","path","TFLOP/s");

    double t1 = run([&](float*o){ k_f16f16<<<gridBlocks,BLOCK>>>(o); }, gridBlocks, 2.0*16*8*16);
    printf("%-26s %12.1f\n","FP16 in / FP16 acc", t1);
    double t2 = run([&](float*o){ k_f16f32<<<gridBlocks,BLOCK>>>(o); }, gridBlocks, 2.0*16*8*16);
    printf("%-26s %12.1f\n","FP16 in / FP32 acc", t2);
    double t3 = run([&](float*o){ k_e4m3f32<<<gridBlocks,BLOCK>>>(o); }, gridBlocks, 2.0*16*8*32);
    printf("%-26s %12.1f\n","FP8 e4m3 / FP32 acc", t3);
    double t4 = run([&](float*o){ k_e4m3f16<<<gridBlocks,BLOCK>>>(o); }, gridBlocks, 2.0*16*8*32);
    printf("%-26s %12.1f\n","FP8 e4m3 / FP16 acc", t4);
    double t5 = run([&](float*o){ k_mxf8<<<gridBlocks,BLOCK>>>(o); }, gridBlocks, 2.0*16*8*32);
    printf("%-26s %12.1f   <== MXFP8 block-scaled\n","MXFP8(e4m3+ue8m0)/FP32acc", t5);

    printf("\n--- ratios ---\n");
    printf("MXFP8-blockscaled / FP16-F32acc = %.2fx   (<== REAL MXFP8 attention vs bf16 attention)\n", t5/t2);
    printf("MXFP8-blockscaled / plain-FP8-F32acc = %.2fx   (block-scaled path escapes the f32 throttle)\n", t5/t3);
    printf("FP16-F16acc / FP16-F32acc = %.2fx   (throttle factor on FP32 accumulate)\n", t1/t2);
    printf("FP8-F32acc  / FP16-F32acc = %.2fx   (FP8 vs bf16 attention, both FP32 acc)\n", t3/t2);
    printf("FP8-F16acc  / FP16-F32acc = %.2fx   (max if FP16 accumulate were usable)\n", t4/t2);
    printf("FP8-F16acc  / FP8-F32acc  = %.2fx   (does FP8 also escape throttle w/ f16 acc?)\n", t4/t3);
    return 0;
}
