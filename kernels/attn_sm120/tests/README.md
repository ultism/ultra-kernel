# MXFP8 Attention Backward — dV+dK 融合 kernel (dvdk2) 数据流

生产路径：`s3b_dvdk2_kernel.cuh` 中 `dvdk2_kernel<false>`（non-WS，256 同质线程，255 regs/thread，warp0 内联发 TMA）。
torch 侧经 `mxfp8_dvdk2_launch` / `use_dk_ws == 3` 启用。

## 任务划分

```
grid = (S/128, H)                        # 每个 CTA 拥有 (一个 kv tile, 一个 head)
accV[128kv,128d], accK[128kv,128d]       # 常驻寄存器，64+64 fp32/thread
循环 m = 0 .. S/64-1                     # 扫过全部 q tile (kBlockM=64)
epilogue: accK/accV 直接 STG -> dK/dV    # 无 atomic（kv tile 独占）
```

## 每步 (q tile m) 数据流

```
                          ┌──────────────────── GMEM ─────────────────────┐
                          │ Q/dO [H,S,128] fp8   Qt/Dt [H,128,S] fp8      │
                          │ SFQ/SFD, SFQt/SFDt   lse/delta [H,S] fp32     │
                          └───────┬───────────────────┬──────────┬────────┘
                                  │ TMA (2-stage)     │ TMA      │ LDG
                                  │ PipeQD            │ (1-stage)│ (c15 预取)
                                  ▼                   ▼ PipeTT   ▼
   ┌──────────── SMEM (86KB / 99KB optin，余 13KB) ────────────────────────┐
   │ sK sV [128,128] fp8 常驻   sQ sD ×2stage   sQt sDt ×1stage   SF ×环  │
   └───────┬───────────────────┬──────────────────┬──────────────────────┘
           │ LDSM(A)           │ LDSM(B, ×8 复制) │ LDSM(B)
           ▼                   ▼                  │
  ① S'  = K·Qᵀ   (mma128, mxfp8 zip, K=128 → 4 k-atom)  → accS  [128kv,64q]
  ② dP  = V·dOᵀ  (同上)                                → accDP [128kv,64q]
           │ release QD stage; 发 QD tile m+2
           ▼
  ③ 量化相位 (纯寄存器, warp 锁步):
       P   = exp2f((S'·sm_scale − lse)·log2e)          # lse/δ 经 shfl 分发
       dS  = P ∘ (dP − δ)                               #   (LSE-dist, 2+2 regs)
       ses = bit-trick ceil(log2(amax32)) − 8           # 无 MUFU (c15)
       P'  = e4m3(P · 2^8)          (const scale)  → tOrP  (A-frag, shfl_fill)
       dS' = e4m3(dS · 2^-ses)      (dynamic)      → tOrDS (A-frag, shfl_fill)
           │ wait TT stage; SF 字节 gather + ×32 填充
           ▼
  ④ 输出 gemm (c14 融合单循环, 2 k-atom):
       LDSM Dt,Qt  ──┐
       accV += P' ·Dt   (mma64 zip: A 数据+SF 成对)
       accK += dS'·Qt
           │ release TT; 发 TT tile m+1
           ▼  下一 q tile
```

## 时序意图（每条 q tile）

```
TMA QD(m) 在 step m−2 发出 ─┐  TMA TT(m) 在 step m−1 (上一步④之后) 发出 ─┐
                            ▼                                            ▼
step m:  [①② gemm burst 32×mma128] → [③ 量化 ~ALU/MUFU] → [④ gemm 16×mma64]
              tensor 忙              tensor 空闲(气泡)         tensor 忙
              ↑ lse/δ LDG 在 ① 前发出，由 32 个 mma 的延迟藏住 (c15)
```

## 已知结构性上限（见 docs/draft.md c16–c20）

- `mma.sync` 无 smem 描述符路径 → B 操作数每 warp 复制 ×8（LSU 56.7% 为固有）
- 8 warp 由共享 barrier 锁步 → ③ 的 tensor 气泡无法用 warp 内重排填掉
- pingpong 需 +64 regs（>255）封死；smem 86/99KB（余 13KB）限制管道加深
- 实测：tensor pipe 60.5%，DRAM 6%，bank conflict 0.75%

## SMEM 明细（dvdk2，sizeof 实测）

| 项 | 大小 | 说明 |
|---|---|---|
| sK + sV | 32KB | kv tile 常驻 [128,128] fp8 ×2 |
| sQ + sD | 32KB | QD 管道 [64,128] fp8 × 2 stage |
| sQt + sDt | 16KB | TT 管道 [128,64] fp8 × 1 stage |
| SF ×6 | 4KB | K/V 512B + Q/D 512B×2 + Qt/Dt 512B |
| sLse + sDlt | 1KB | [2][64] fp32 ×2 |
| mbarrier | 48B | PipeQD + PipeTT |
| **合计** | **86.0 KB** | optin 上限 99KB → 余 13.3KB |

## 变体与基准

| kernel | 线程 | 说明 |
|---|---|---|
| `dvdk2_kernel<false>` | 256 | **生产**。无 WS，warp0 全 warp 当 producer（elect 发 TMA） |
| `dvdk2_kernel<true>` | 384 | WS 版（WG0 producer @24 regs），实验中略慢，保留对照 |

## WS 变体 (`dvdk2_kernel<true>`) 数据流

384 线程 = 1 个 producer WG + 2 个 consumer WG，寄存器按角色重分配
（`warpgroup_reg_dealloc/alloc`）：

```
WG0 (128 thr)                     WG1, WG2 (256 thr = NumMma)
warpgroup_reg_dealloc<24>()       warpgroup_reg_alloc<232>()
┌─────────────────────────┐      ┌──────────────────────────────────┐
│ warp0: producer_acquire │      │  与 non-WS 完全相同的 step(m):     │
│   elect lane 发 TMA:    │      │   ① S'=K·Qᵀ  ② dP=V·dOᵀ           │
│   QD(m)+TT(m) 每条 q tile│ ───▶ │   ③ 量化 → P'/dS'                 │
│   for m in 0..MT-1      │ 2条  │   ④ accV+=P'·Dt  accK+=dS'·Qt     │
│   (两条管道同环推进)      │ 管道  │  tid = threadIdx.x − 128          │
│ 发完 return             │      │  epilogue: STG dK/dV              │
│ (warp1-3 立即 return)    │      └──────────────────────────────────┘
└─────────────────────────┘            ▲ NamedBarrier(NumMma,1) 等 K/V 常驻就位
        PipeQD 2-stage / PipeTT 1-stage 角色: Producer | Consumer
```

与 non-WS 的唯一区别在 **谁发 TMA**：WS 用独立 WG（寄存器释放到 24，consumer 扩到
232），non-WS 由 warp0 在 step 间隙内联发出（ProducerConsumer 角色，consumer 仍是
255 regs 全程）。non-WS 里 producer 必须是**全 warp 会合**进 `producer_acquire`
（elect 单 lane 发 copy）—— 若让单 lane 分叉跑在前面，会和本 warp 的 LDSM/shfl
乱序。实测 WS 版不占优：producer WG 的 barrier 会合开销 + consumer 少 23 regs
（232 vs 255，量化相位更紧），见 `docs/draft.md` c-系列记录。

## 共享初始化（两种变体相同）

```
sK/sV [128,128] fp8 + sSFK/sSFV 512B   ←  UniversalCopy<uint128_t> 一次性 LDG->SMEM
lse/delta [H,S] fp32                    ←  每 step 2+2 regs LDG (c15: gemm burst 前发出)
SFQ/SFD/SFQt/SFDt                       ←  随 TMA 管道进 SMEM, 用前 LDS/字节 gather
```

验证：`test_mxfp8_bwd_dvdk2.py`（fp64 ref；mode3 与 mode1 逐位一致 @ H4/S4096 与 H32/S16896）。
harness：`s3b_ws_e2e.cu`（`-DS3B_BENCH` 跑 H32/S16896）。

## dQ kernel (`dq_ws`) — 生产 WS 版

dQ 与 dvdk2 的驻留轴对偶：**q 驻留**（每个 CTA 拥有一个 q tile，扫全部 kv），
所以 dQ 必须走 WS —— KV/Kt 两条管道随 kv 环流，TMA 量大且规则。

```
grid = (S/128, H)                       # CTA 拥有 (q tile, head); kBlockN=64 (kv)
accQ[128q,128d] 常驻寄存器               # q tile 独占 -> epilogue 直接 STG, 无 atomic

WG0 (128 thr) producer @24 regs         WG1,WG2 (256 thr) consumer @232 regs
┌────────────────────────────┐         ┌─────────────────────────────────┐
│ warp0 elect: for n in NT:  │         │ 每 kv step n:                    │
│   PipeKV(2-stage): K,V,    │ ──────▶ │  ① S' = Q·Kᵀ   (mma128, A=Q 常驻) │
│     SFK,SFV                │         │  ② dP = dO·Vᵀ  (mma128, A=dO 常驻)│
│   PipeKt(2-stage): Kt,SFKt │ ──────▶ │  ③ P=exp2f(...), dS=P∘(dP−δ),    │
│                            │         │     动态 amax -> dS' -> tOrDS     │
└────────────────────────────┘         │  ④ accQ += dS'·Kt (mma64)        │
                                        └─────────────────────────────────┘
sQ sD [128,128] fp8 常驻 SMEM（S'/dP 的 A 操作数）; sK sV sKt 走管道
```

关键结构差异（对偶于 dvdk2）：

| | dvdk2 (kv 驻留) | dq_ws (q 驻留) |
|---|---|---|
| 常驻 SMEM | sK, sV | sQ, sD |
| 管道内容 | Q/dO (QD) + Qt/Dt (TT) | K/V (KV) + Kt (Kt) |
| 累加器 | accV+accK (128×128 ×2) | accQ (128×128) |
| 量化产物 | P' + dS'（两份 A-frag） | dS' 一份 |
| 输出 gemm 的 B | Qt/Dt（转置 tile，1-stage） | Kt（2-stage） |
| WS 必要性 | 无（non-WS 更快，生产用） | 有（TMA 吞吐大，producer 专职） |

dS' 量化经 fwd 的 S5 intra-quad `__shfl` 直接进 A-frag 布局（无 sDS smem
往返、无 NamedBarrier）；动态 per-32 amax 保留（与 dvdk2 同一套 bit-trick）。
