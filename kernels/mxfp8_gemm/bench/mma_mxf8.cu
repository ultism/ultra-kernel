// MXFP8 block-scaled MMA throughput on sm120a.
// Instruction: mma.sync...kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32...f32...ue8m0
#include <cstdio>
#include <cuda_runtime.h>
#ifndef NACC
#define NACC 8
#endif
#ifndef ITERS
#define ITERS 120000
#endif
#define BLOCK 256
#define CK(x) do{ cudaError_t e_=(x); if(e_){printf("CUDA err %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(e_)); exit(1);} }while(0)

__global__ void k_mxf8(float* out){
    unsigned a0=0x38383838,a1=0x38383838,a2=0x38383838,a3=0x38383838; // e4m3 ~1.0
    unsigned b0=0x38383838,b1=0x38383838;
    unsigned sfa=127u, sfb=127u;            // ue8m0 byte 127 -> 2^0 = 1.0
    unsigned short bidA=0,tidA=0,bidB=0,tidB=0;
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
                "r"(sfa),"h"(bidA),"h"(tidA),
                "r"(sfb),"h"(bidB),"h"(tidB));
        }
    }
    float acc=0;
    #pragma unroll
    for(int j=0;j<NACC;j++) acc += c[j][0]+c[j][1]+c[j][2]+c[j][3];
    if(acc==-1.f) out[threadIdx.x]=acc;
}

int main(){
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p,0));
    int sm=p.multiProcessorCount, gridBlocks=sm*4;
    float* o; CK(cudaMalloc(&o,BLOCK*sizeof(float)));
    k_mxf8<<<gridBlocks,BLOCK>>>(o); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    cudaEvent_t s,e; CK(cudaEventCreate(&s)); CK(cudaEventCreate(&e));
    float best=1e30f;
    for(int r=0;r<3;r++){
        CK(cudaEventRecord(s)); k_mxf8<<<gridBlocks,BLOCK>>>(o);
        CK(cudaEventRecord(e)); CK(cudaEventSynchronize(e));
        float ms; CK(cudaEventElapsedTime(&ms,s,e)); if(ms<best)best=ms;
    }
    double warpMMAs=(double)gridBlocks*(BLOCK/32)*NACC*(double)ITERS;
    double tflops=warpMMAs*(2.0*16*8*32)/(best/1e3)/1e12;
    printf("MXFP8 block-scaled e4m3/ue8m0 / FP32 acc : %.1f TFLOP/s  (sm_%d%d, %d SMs)\n",
           tflops, p.major, p.minor, sm);
    return 0;
}
