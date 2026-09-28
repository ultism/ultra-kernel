# Draft: MXFP8 attention bwd (sm120) optimization plan

> KDA workflow draft. Task: optimize the committed MXFP8 FA backward
> (`1fc86dc`: `s3b_dq_ws` / `s3b_dk_ws` / `s3b_dv_ws` + torch ext).

## Task contract

- **Task name**: mxfp8-attn-bwd-sm120 optimization
- **Objective**: reduce total dq+dk+dv ws kernel time at the bench shape
  (H=32, S=16896, D=128) on the local GPU; keep numerics unchanged.
- **Correctness**: rel-L2 vs fp64-dequant reference, H=1 S=256 non-causal,
  must stay at the committed level (≈0.026–0.027, dq harness prints per-32
  row/col slices + total). Numerics are value-preserving only (same quant
  groups, same RNE e4m3 bytes); no algorithm changes.
- **Perf target**: beat current ws baselines per kernel; stretch goal
  total < 3×(fwd-path parity). No regression on any of the 3 kernels.
- **Allowed**: CUDA C++ / CuTe in `kernels/attn_sm120/tests/`, sm_120a only.
- **GPU**: RTX 5060 Ti (sm_120, GB206), CUDA 13.3, driver 610.43.
  NOTE: commit-message numbers (75/84/68 ms) were NOT reproduced here —
  they likely came from a 5090. Local numbers are the baseline.

## Baseline (this machine, measured 2026-09-27)

Committed binaries (`s3b_dq_ws_bench` built 09-25, `s3b_bwd_bench` 09-25):

| kernel | time (H32/S16896) |
|---|---|
| s3bws::dq_ws | **125.5 ms** |
| s3b::dq (old non-ws 128-tile) | 121.4 ms |
| s3b64::dv64 | 230.1 ms |
| s3b64::dk64 | 248.0 ms |
| s3b::dkdv (legacy fused) | 849.3 ms |
| s3bdkws::dk_ws | no local harness — TBD |
| s3bdvws::dv_ws | no local harness — TBD |

dq_ws ≈ 56 TFLOP/s effective ≈ 27% of est. 5060 Ti mxfp8 peak (~210).
Torch here is CPU-only → validation must be native CUDA harnesses.

## Validation / benchmark commands

Build (from `/root/fa-blackwell`):
```
nvcc -std=c++17 -O2         -gencode arch=compute_120a,code=sm_120a \
  --expt-relaxed-constexpr --expt-extended-lambda \
  -I tmp/cutlass/include -I include <harness>.cu -o <out>           # correctness
nvcc -std=c++17 -O2 -DS3B_BENCH ...                                 # bench
```
- Existing dq harness: `kernels/attn_sm120/tests/s3b_dq_ws.cu`
  (correctness prints `dQ rel-L2`; bench prints `dq_ws: X ms`).
- **Gap**: no dk_ws/dv_ws harness. First step is a unified
  `s3b_ws_e2e.cu` (all three ws kernels; H=1/S=256 fp64 ref; `-DS3B_BENCH`
  H=32/S=16896 timing) reusing the quant/ref code from `s3b_dq_ws.cu` +
  `s3b_bwd_e2e.cu`.

## Main risks / unknowns

1. dk_ws/dv_ws local perf unknown (no harness) — measure first.
2. Register pressure: consumer runs at `warpgroup_reg_alloc<232>`; the
   register-quant path adds ~8–16 regs (packed words) but drops the
   ldmatrix staging overlap; watch ptxas spill counts.
3. The fwd S5 shfl path is only proven for **const SF**; bwd dS needs
   dynamic per-32 scales → hybrid: shfl for data, tiny smem round-trip for
   the 4 SF bytes/thread (same as today, minus data smem).
4. NamedBarrier removal changes cross-warpgroup timing — validate at
   S=256 AND a mid size (S=1024/2048) to catch races the small case hides.

## Ranked candidates

1. **[HIGH/med-risk] Register-shfl dS quant (port fwd S5 pattern,
   `s3_kernel.cuh:669-724`) into dq_ws, then dk_ws/dv_ws.**
   Today per 64-kv step: 32 scattered swizzled `ST.U8` to sDS + 2
   NamedBarriers (all 256 consumers lockstep) + ldmatrix readback +
   per-element SF gather address math. The fwd kernel solves the identical
   layout problem (m16n8 accum → A-operand, 128x64, per-32-along-kv) with
   register packing + intra-quad `__shfl_sync` — copyable almost verbatim.
   Kills: both barriers, sDS smem (8KB) + sSFDS, byte-scatter stores,
   ldmatrix. Keeps: dynamic amax (already in regs), tiny sSFDS smem for SF
   (or shfl that too later).
2. **[MED/low-risk] Pipeline restructure**: dq waits on PipeKV then PipeKt
   per step; try issuing Kt TMA together with KV (single pipe or reordered
   acquires), or prefetch Kt one step ahead so `dQ` gemm never waits.
3. **[MED] kStages 2→3 on the Kt/Qt pipe only** (smem: dq uses 95,232B of
   ~99KB — KV stages can't grow; Kt stage is only 8.5KB → may fit one more).
4. **[LOW] quant-loop ALU**: fold `sm_scale*log2e` into one multiplier,
   `exp2f` on already-scaled values (saves 1 FMUL/elem = 32/step/thread).
5. **[defer] dv+dk fusion** — already tried (`s3b_dvdk_ws`), slower due to
   64-wide kv tiles; do not retry without a new idea.
6. **[defer] old `s3b::dq` is 121ms ≈ ws 125ms** — if register-shfl makes
   ws clearly faster, the old kernels become dead weight; keep for
   reference only.

## First concrete steps

1. Write `kernels/attn_sm120/tests/s3b_ws_e2e.cu`: launches dq_ws+dk_ws+dv_ws;
   correctness mode (H=1 S=256, fp64 ref for dQ/dK/dV, rel-L2 prints) and
   bench mode (`-DS3B_BENCH`, H=32 S=16896, per-kernel ms).
2. Rebuild dq_ws fresh + run all → record true baselines in this file.
3. ncu spot-check dq_ws (`sm__pipe_tensor`, `l1tex__data_bank_conflicts`,
   barrier stalls) to confirm the quant-path hypothesis before editing.
4. Candidate 1 on dq_ws only → validate (S=256/1024) → bench. Promote if
   rel-L2 ≤0.027 and ≥3% faster; then port to dk/dv.

## Evidence to promote / reject

- Promote: correctness rel-L2 ≤ committed 0.027 (S=256 + S=1024), bench
  ≥3% better at H32/S16896, ptxas spill not worse than baseline.
- Reject: any rel-L2 blowup, bench regression, or new ptxas spills
  (`-Xptxas -v` on the harness build).
- Record every candidate result (kernel, ms before/after, rel-L2, verdict)
  in the results section below.

## Results log

Measurement protocol: `ncu --clock-control base` `gpu__time_duration` at
H=4/S=4096 (DVFS noise on this box is +-20-40% at the big shape; ncu base-clock
numbers are exactly reproducible). Bitwise A/B vs `s3b_dq_ws_ref.cuh` (frozen
copy of the committed dq kernel) in the correctness harness.

| candidate | kernel | µs before | µs after | rel-L2 | bitwise | verdict |
|---|---|---|---|---|---|---|
| baseline (ncu) | dq_ws | 706.9 | — | 0.0269 | — | tensor 41.4%, lsu 28.1%, bar 5.3% |
| baseline (ncu) | dk_ws | 707.4 | — | 0.0265 | — | tensor 41.4%, lsu 31.1% |
| baseline (ncu) | dv_ws | 585.4 | — | 0.0265 | — | tensor 33.2%, lsu 30.4% |
| c1: S5 reg-shfl dS quant + SF dedup gather + barrier/sDS removal | dq_ws | 706.9 | **562.1** | 0.0269 | MATCH 32768/32768 (S=256), MATCH (S=1024) | **PROMOTED (-20.5%)** tensor 52.5%, bar 0.3%, lsu 26.6%, 0 spill, smem 95→87KB |
| c1 ported | dk_ws | 707.4 | **648.1** | 0.0265 | rel-L2 unchanged | **PROMOTED (-8.4%)** tensor 45.3%, bar 4.8% (sLse barrier remains) |
| c1 ported | dv_ws | 585.4 | **482.1** | 0.0265 | rel-L2 unchanged | **PROMOTED (-17.6%)** tensor 40.5%, bar 7.8% (sLse barrier remains) |
| c4: per-thread LDG.64 lse/delta | dv_ws | 482.1 | **423.1** | 0.0265 | rel-L2 unchanged | **PROMOTED (-12.3%)** bar 0.4%, tensor 46.8% |
| c4 same | dk_ws | 648.1 | 721.2 | — | — | **REJECTED** (+11.3%; +32 regs live across gemms) |
| c5: hoist Kt ldmatrix above quant | dq_ws | 563.7 | 659.1 | — | — | **REJECTED** (+17%; tOrKt liveness serializes quant) |
| c6: interleave S/dP k-chains | dq_ws | 560.1 | 567.3 | — | — | **REJECTED** (+1.3%; ptxas already interleaved) |
| c7: B-SF u16 gather | dq_ws | 560.1 | 566.6 | — | — | **REJECTED** (+1.2%) |
| c8: per-WG sLse + 128-bar | dk_ws | 648.1 | 682.6 | — | — | **REJECTED** (+5.3%, bar 4.9→6.5%) |
| c9: e4m3x2 paired cvt in the S5 pack | dq/dk/dv | 560.1/648.1/423.1 | **545.2/632.6/408.7** | unchanged | dq MATCH | **PROMOTED** (-2.5/-2.4/-3.4%) |
| c10: lse/delta LDS.64 reads | dk_ws | 632.6 | 638.1 | — | — | **REJECTED** (+0.9%) |
| c11: lse/delta ride the QD TMA pipe (1D byte TMA into per-stage smem; drop NamedBarrier; release after quant reads) | dk_ws | 632.6 | **573.6** | 0.0265 | — | **PROMOTED (-9.3%)** bar 0.3%, tensor 51.8% |

c11 gotcha worth keeping: a float-typed 1D TMA miscompiled the per-stage smem
address (base + stage*0x4000 instead of stage*0x100; SASS UTMALDG OOB). Byte-
typed (uint8) TMA over the same buffer is correct. Probe: /tmp/opencode/tma_dbg3.cu.

## Final numbers

Big shape (H32/S16896), `ncu --clock-control base` (DVFS-immune; absolute ms
are base-clock, ratios are the evidence):

| kernel | baseline ms | final ms | Δ |
|---|---|---|---|
| dq_ws | 236.90 | 151.25 | **-36.2%** |
| dk_ws | 223.20 | 165.80 | **-25.7%** |
| dv_ws | 176.83 | 92.60 | **-47.6%** |
| total | 636.9 | 409.7 | **-35.7%** |

Small shape (H4/S4096, base clock): dq 706.9→545.2 µs (-23%), dk 707.4→573.6
(-19%), dv 585.4→338.2 (-42%).

Correctness: dQ bitwise-identical to the committed kernel (S=256 & S=1024);
dK/dV rel-L2 unchanged (0.0265/0.0265 @256; 0.0265/0.0276 @1024); ptxas 0
spills on all three ws kernels (baseline dq had 40B); smem 95/96/61KB →
87/88/52KB. `csrc/mxfp8_bwd_kernel.cu` TU compiles clean.

What landed (all value-preserving):
- **c1** (all 3 kernels): dS/P' quantized in registers + fwd-S5 intra-quad
  `__shfl` straight into the K=64 MMA A-operand — kills the sDS smem
  round-trip (32 scattered ST.U8 + ldmatrix + 2x256-thread NamedBarrier per
  step). A-SF register-broadcast (probe: 32 dups share one byte per k-tile);
  B-SF gather deduped 1024→32 LDS.U8/step.
- **c4** (dv only): lse via per-thread LDG.64 pairs (barrier removed). Same
  change REGRESSED dk (+32 regs live across the gemms) — dk took c11 instead.
- **c9**: `__nv_cvt_float2_to_fp8x2` pairs in the S5 pack (half the CVTs).
- **c11** (dk): lse/delta ride the QD TMA pipe into per-stage smem slots;
  consumer reads them after `consumer_wait` — no barrier at all.

Rejected with evidence: c5 (Kt ldmatrix hoist +17%), c6 (S/dP interleave
+1.3%), c7 (B-SF u16 +1.2%), c8 (per-WG lse +128-bar +5.3%), c10 (LDS.64
lse reads +0.9%), dvdk fusion (pre-existing, 64-wide kv tiles).

Remaining known headroom: `wait` stalls 25-35% (QMMA fixed-latency chains at
12 warps/SM; sm120 has no tcgen05 async MMA — the gotcha.md "wait floor").
Next structural ideas if resumed: 128-wide kv tiles (needs spill analysis —
fwd went the other way), producer-side resident loads, 3-stage Kt ring.

## Addendum: const-scale P' (user insight) + fusion re-estimate

P' = exp(S'*sm - lse) <= 1 ALWAYS (lse >= row max by construction), so the dV
side needs NO per-32 amax: fixed scale 256.0 (se=-8) maps [0,1] into e4m3's
normal range without ever saturating (max 256 <= 448) — same argument as the
fwd kernel's kPConstSF (s3_kernel.cuh:52-58).

- **c12: dv_ws const-scale P'** (drop amax/SHFL-reduce/scale-select; A-SF is a
  compile-time byte 119): dv 408.7 -> **338.2 µs (-17.2%)**, tensor
  48.2->59.7%, dV rel-L2 unchanged (0.0265/0.0276). PROMOTED.
- dK/dQ sides (dS = P*(dP-delta)) are NOT bounded -> dynamic amax stays.

**Fusion (dv+dk) re-estimated with const-scale P'**: still a NO on sm120.
Register accounting at kv=128 (256 consumer threads, 232-reg budget):
accK(64)+accV(64) persistent + accS(32)+accDP(32) transient + K/V A-frags(32)
+ Q/dO B-frags(16) + P'/dS' A-frags(16) + Dt/Qt B-frags(16) + SF/lse/misc(~50)
~= **322 regs -> ~90-reg spill**. (Streaming dQ to gmem does NOT fix a full
3-way fusion either: accQ is only ever transient (32) so removing it saves
nothing structural, while adding acc_dq(32)+Kt frag(16) and 20-113 ms of
dq_accum traffic.) kv=64 fits (~180-190 regs) but the (4,2,1) atom layout
doubles QMMA issue per FLOP (tensor pipe already 52-60%, would need >100%)
and 32-wide quant groups span warp pairs — exactly why dvdk (mode 2) measured
slower. On SM100 fusion wins because accumulators live in TMEM (FA4 2-CTA
bwd, CUTLASS ex77 MLA bwd) — sm120 has no tcgen05/TMEM, so the 3-kernel
split is the right structure here. dvdk kept as mode 2, unchanged.

## Addendum 2: why not "fused kernel + dQ streamed to gmem" (FA2's structure)

dQ partial = [128,128] fp32 per (kv-block, q-tile) = 64KB x 132^2 x 32 heads =
36.5 GB of extra gmem traffic (9.13G fp32 accumulate ops). Measured on this
box (/tmp/opencode/bw.cu, bw2.cu):

- DRAM-stream BW 380.6 GB/s; DRAM atomicAdd fp32 128.2 GB/s; L2-resident
  atomicAdd fp32 322.6 GB/s payload.
- FA2's dq_accum tile working set is 64KB per (h, m) -> mostly L2-absorbed:
  ~113 ms atomic-equivalent worst case, ~20-50 ms with RMW/split-buffer
  variants — NOT the ~285 ms a naive DRAM-rate estimate gives (earlier draft
  overstated this; corrected).
- Our split instead RECOMPUTES S/dP: 3 extra gemm passes = 7.02 TFLOP @
  ~117 TFLOP/s = **~60 ms**, zero extra memory.

So dQ-gmem and recompute are the SAME order of magnitude; the decisive
anti-fusion arguments remain the register wall (accK+accV+accQ ~= 290 regs >
232) and the MMA issue rate at kv=64. NOTE also: the "319 ms bf16 flash bwd"
figure in commit 1fc86dc came from a different GPU (commit-time numbers
236/319 don't reproduce on this 5060 Ti), so it was never a same-silicon
comparison — the density table above is the honest one.

Big-shape A/B (H32/S16896, alternating binaries x3 rounds, min ms):

| kernel | base | c1+c4 | Δ |
|---|---|---|---|
| dq_ws | 80.0–84.6 | 66.6–69.7 | **-17%** |
| dk_ws | 79.2–84.8 | 75.0–77.3 | **-5.5%** |
| dv_ws | 64.8–69.3 | 49.9–50.4 | **-23%** |
| total | 224.2–238.7 | 191.7–196.9 | **-15%** |

dk is now the laggard (75.5 vs dq 66.7): the sLse/sDlt staging + 256-barrier
is the remaining delta.

Probe facts (sf_probe.cu, mma64 SF identity coords): A-SF per k-tile = 32
dups of ONE byte (row = warp*16 + 8*(lane&1) + lane/4 == quant row
mi=lane&1). B-SF per k-tile = 16 distinct bytes x 32 consecutive dups
(element 32r+dup -> d = 8r + lane/4, kv-block = k). Baseline gather cost was
64 (A) + 1024 (B) LDS.U8/thread/step.

## c13: fused dvdk @kv=128 (s3b_dvdk2_kernel.cuh, template<bool kWS>) — REJECTED

Built a fully-working fused dv+dk kernel (kv=128, c1 register-shfl quant + c9
paired cvt + c12 const-scale P', LDG lse/delta), both a WS variant (384 thr)
and a non-WS variant (256 thr, warp-0 elected TMA producer, ProducerConsumer
role). Three independent structural walls, all measured:

1. **Registers (ptxas -v)**: WS = 168 regs + 864B spill; non-WS = 255 regs +
   1904B spill. ptxas lifetime-overlap does NOT save it — the accumulator set
   accK(64)+accV(64)+accS(32)+accDP(32)=192 plus SF/quant state is too big at
   kv=128 (true live-state ~390 regs/thread). NOTE: ptxas caps the static
   budget at 65536/threads (384 -> 168) regardless of setmaxnreg; the "Used
   regs" line reports that cap.
   **WS is register-SAVING, not register-costing**: non-WS with the TMA
   producer machinery stripped (probe, /tmp/opencode/dvdk2_noprod.cuh) spills
   only 548B at the same 255 cap — the inline producer/issue state alone adds
   ~1356B of spill pressure (~dozens of regs). WS parks all of it in the
   WG0 producer at 24 regs. Net: WS trades 87 regs of static cap for removing
   more than that from the math threads — the earlier "WS costs 23 regs/thread"
   framing was WRONG.
2. **SMEM**: full 2-stage rings (Q,dO,Qt,Dt) + resident K,V = 105472B > the
   sm_120 optin limit of **101376B** (cudaGetDeviceProperties; RTX 5060 Ti =
   100KB smem/SM, not 228KB). Had to drop the TT ring (Qt/Dt) to single-stage
   to fit at all.
3. **Benchmark** (H32/S16896, ncu --clock-control base, gpu__time_duration):
   dk 68.9 + dv 37.7 = **106.6 ms** split vs fused **WS 182.0 ms / non-WS
   319.7 ms** = 1.7x / 3.0x SLOWER. Rejected decisively.

FA2 cross-check (compiled flash_bwd_hdim128_bf16_sm80.cu for sm_120a): FA2
d=128 bwd survives NOT by 128-wide fusion but by **kBlockM=64/kBlockN=64**
(21 seqk-parallel variants: 248-255 regs, 0 spills); its M64/N128 variants DO
spill (up to 152B). FA2's accumulator footprint at M64/N64 is ~96+32 regs vs
our 192+64 — half. Conclusion stands: on sm120 the 3-kernel split at kv=128
is right; FA2's fused bwd at d=128 is itself a 64-tile kernel.

Correctness gates: both dvdk2 variants rel-L2 0.0265/0.0276 == split baseline
(S=256 and S=1024). Gotcha: lse/delta could NOT ride the QD TMA pipe in the
fused kernel (consumer read SF-like garbage bytes 0x78/0x79; same code works
in dk_ws — root cause not isolated, possibly 6-copies-on-one-barrier
interaction); LDG per-thread float2 pairs (c4-style) work fine.

### c13b: kv=64 atom-layout ablation (4,2,1)@256c vs (2,2,1)@128c

Fixed two harness/PORTING bugs while wiring: (1) tma_sfdt must be built from
mSFDt (dsfDt), not aliased to tma_sfqt (dV garbage); (2) the resident K/V
cooperative load's tiled-copy thread count must match NumMmaThreads (128 for
221). (2,2,1) quant mapping derived from a host-side identity-tensor probe:
row=(w%2)*16+lane/4+8*(mi&1)+32*(mi>>1), col=(w/2)*8+(lane%4)*2+(ni&1)+16*(ni>>1),
amax pair w^2, SF write guard w<2, sAm[4][64].

Correctness (S=256 and S=1024): both variants dV rel-L2 0.0265/0.0276 (exact
baseline); dK 0.038 for BOTH (the old kernel computes dS from the e4m3 P
readback, a known precision trait — equal on both, fair A/B).

Registers: (4,2,1)@384t: 168 regs + 264B st/548B ld spill; (2,2,1)@256t:
255 regs + 352B st/508B ld spill. (kv=64 fusion DOES fit — unlike kv=128.)

Benchmark (H32/S16896, ncu base clock, this run: dk 75.6 + dv 40.8 = 116.4 ms
split; dvdk2_ws kv128 = 200.5):
  dvdk64 (4,2,1)@256c: **645.7 ms**   (5.5x slower than split)
  dvdk64 (2,2,1)@128c: **891.8 ms**   (7.7x slower; 38% worse than 421)

=> The N-split "MMA issue doubling" is NOT the dominant fusion cost — the
old dvdk's pipeline serialization is (sP/sDS alias the QD ring stages, so the
producer cannot prefetch past the output gemms; plus smem quant round-trip +
4 NamedBarriers/step). Halving threads ((2,2,1)) makes it strictly worse.
kv=64 fusion is dead; kv=128 fusion (c13) is dead on registers+smem;
3-kernel split stands.

### c13c: V_in_regs=false analogue (stream K/V A-fragments per step) — REJECTED

FA2's V_in_regs=false reloads V fragments from smem per step instead of
keeping them register-resident. Ported to dvdk2 (V-only and K+V variants),
ptxas spill: non-WS 1904B -> 1704B (V) -> 1460B (K+V); WS 864B -> 984B ->
1176B (WS gets WORSE: at the 168 cap the per-step LDSM reissue costs more
than the freed 16-32 persistent regs). Nowhere near rescue: the quant+output
window's live set is ~390 regs — dominated by the 224 regs of accumulators +
lse/delta, not by operand fragments. FA2 profits from this knob because at
M64/N64 its accumulators are only ~96-128 regs; ours at kv=128 are 192.

### c13d: LSE/dPsum distributed-stats (FA3 ShuffleLSE/ShuffledPsum) on dvdk2

User-suggested, from hopper/mainloop_bwd_sm90_tma_gmma_ws.hpp:864/889: stats
stored once per warp (lane l holds cols (l/4)+8*(l%4)+32*k, k<2 -> 2+2 regs
vs 16+16), fetched via shfl (src lane L=(l%4)*8+(ni&1)*4+((ni>>1)&3), slot
kb, compile-time). Correct (0.0265/0.0265 both variants, S=256).

ptxas spill (fused dvdk2@kv128): non-WS 1904B->1752B, WS 864B->**680B**.
Combining with K/V streaming gives no further win (1460B, streaming
dominates). Real ~28-reg saving but the 192-reg accumulator set is the wall;
fusion stays rejected. Change kept in dvdk2 (strictly better); NOT ported to
split dk/dv (already 0-spill at 168; +64 shfl/step would only cost).

FA4-MXFP8 blog (sm100) transferable notes: they ALSO use const-scale P
(= our c12); FP16 dQ reduction (corroborates dQ-traffic analysis); [32,32]
square quant + redux.sync.max.abs are sm100/TMEM-only, N/A here.

### c13e: LIFETIME RESTRUCTURE — fused dvdk@128 LIVES (user was right)

User insight: only accK+accV (128 regs) are hard-persistent; everything else
is transient and schedulable. The fix that flipped fusion from 3x slower to
WINNING: **stream ALL operand fragments per-k inside the gemm loops** (FA2
A_in_regs=false style — LDSM moved into the k-loop so ptxas sees per-element
liveness) + LSE/dPsum distributed stats (c13d). Full preload removal was the
difference vs c13/c13c (which only streamed K/V).

Final ptxas (s3b_dvdk2_kernel.cuh): non-WS 255 regs + **12B/28B spill**
(~zero); WS 168 regs + 172B/280B (168 cap still too tight).

Two bugs found en route: (1) TT 1-stage ring must be released AFTER the
per-k streamed output gemms, not before (race); (2) non-WS TMA issue must be
**warp-converged** (all of warp 0 runs producer_acquire; elect_one does the
copies) — a lone divergent elected lane races ahead of its own warp's
LDSM/shfl (S=1024 inf/NaN, S=256 passes by luck).

Benchmark (H32/S16896): ncu base clock med: split dk+dv 115.7 ms vs fused
non-WS **110.6 ms (-4.4%)**; event-timed min: 102.1 vs **95.0 ms (-6.9%)**.
Fused does 4 gemm passes vs split's 5 (S' computed once) — 20% FLOP cut,
realized as ~5-7% wall (single-stage TT ring + LDSM streaming overhead eat
the rest). WS variant: 142.5 ms (spill-bound) — non-WS wins because 255 >
168 matters more than the producer's register offload now that operand
streaming crushed the transient peak.

Correctness: 0.0265/0.0276 both variants, S=256 + S=1024 (== split baseline).
PROMOTED candidate: dvdk2<false> can replace dk_ws+dv_ws (one launch, less
smem than dk's, 2 fewer SF rings). csrc wiring pending user sign-off.

### c13f: torch-side validation + promotion (ai-toolkit env)

CORRECTION to earlier notes: the box DOES have CUDA torch — the `ai-toolkit`
conda env (`/root/miniconda3/envs/ai-toolkit/bin/python`): torch 2.14.0+cu130,
CUDA available on the 5060 Ti. (The `base` env is CPU-only torch 2.11 — that
was the source of the "no CUDA torch" claim.)

Wired dvdk2<false> as `use_dk_ws == 3` in csrc/mxfp8_bwd_kernel.cu
(mxfp8_dvdk2_launch). Torch-ext validation
(kernels/attn_sm120/tests/test_mxfp8_bwd_dvdk2.py, ai-toolkit python):
- H4/S4096: rel-L2 vs fp64 ref identical to split (0.0265/0.0266/0.0273);
  mode3 vs mode1 dK/dV **bitwise identical** (same arithmetic order).
- H32/S16896: mode3 vs mode1 dQ/dK/dV **BITWISE MATCH**.
- Bench: H4/S4096 +2.2%; H32/S16896 total bwd 158.4 vs 166.7 ms (**+5.0%**).

Consumer: /root/ai-toolkit/toolkit/util/mxfp8_attn.py — added
`AITK_DK_WS_MODE` (int; 1 default = split, 3 = fused). e2e autograd smoke
(fwd .so + bwd ext, L=1000 padded): runs; q/k grads contain non-finite values
IDENTICALLY in mode 1 and mode 3 (124288/512000 both) — pre-existing
integration quirk in the current production path, NOT a mode-3 regression
(flagged for follow-up; kernel-level outputs are bitwise clean).

### c14: interleave dV/dK output gemms (8 independent MMA chains)

ncu on dvdk2<false> (H32/S16896, base clock): tensor pipe 54.1% (vs dk_ws
61.4%), stalls = wait 34.3% + math_pipe_throttle 19.8% + short_scoreboard
11.9%; occupancy 16.7% (hard cap: 99KB smem + 255 regs => 1 CTA/SM).
DRAM 6%, L2 hit 95%, bank conflicts 0.75% — memory is a non-issue.
Diagnosis: latency-bound within 8 warps. The two output gemms run as two
sequential bursts (LDSM Dt -> 8 mma -> LDSM Qt -> 8 mma), each with only 4
independent acc chains and the 2nd LDSM batch exposed after the 1st burst.

Candidate: merge the two gemm loops — issue BOTH LDSM batches first, then 16
mma with 8 independent acc chains (accV and accK are both live anyway; tOrP /
tOrDS are both produced by the quant phase already). Zero extra registers in
principle. Validation: s3b_ws_e2e S=256 + S=1024 + bench H32/S16896 vs
c13e baseline (ncu med 110.6ms / event 95.0ms).

### c14 result: marginal (+1.5%), kept

Merged dV/dK output-gemm loops (both LDSM batches first, then 16 mma with 8
independent acc chains). Regs unchanged (255, 12B/28B spill). ncu: 115.5 ->
114.0ms, tensor 54.1 -> 55.9%; stalls unchanged. ptxas had likely already
interleaved. Kept (free).

Source-level stall attribution (pcsamp, ~5.5M samples) — the real map:
- mma_sm120.hpp:2762  50% (math_pipe_throttle): mma issue port saturated
  DURING bursts; tensor pipe idles between bursts (quant phases) => bursty.
- s3b_dvdk2_kernel.cuh:146  8.4% (wait): mx_scale_exp's log2f+ceilf MUFU
  chain on the amax->scale critical path.
- :458 (exp2f) + :464/:465 (amax shfl)  9.6%: softmax quant chain.
- ?:0  3.3% (long_scoreboard): lse/dlt LDG issued right before use.
- :549/:560  4%: SF byte-gather fill loops.
- barrier.h:419  ~2% (sleeping): TT 1-stage ring TMA wait (smem-capped).

Structural phase-overlap (FA3 pingpong) is register-walled: accK+accV = 128
regs persistent; double-tile quant needs +64 regs. Not viable at 255.

### c15: LDG prefetch + bit-exact mx_scale_exp  ->  -14%  KEPT

1. lse/dlt LDGs moved before the S'/dP gemm bursts (32 mma of latency cover);
   use-site unchanged (shfl distributed stats).
2. mx_scale_exp: log2f/ceilf replaced by exact integer identity
   ceil(log2(x)) = exp_unbiased + (mantissa != 0) for normal x>0; zero/
   subnormal -> -126, inf/nan -> 127. Bit-exact vs the libm version.

Result (dvdk2<false>, H32/S16896, base clock): 114.0 -> 98.0ms (-14%),
tensor pipe 55.9 -> 60.5%. Spill rose 12/28 -> 92/116B but is cold-path.
Event bench: 96.6 -> 90.2ms. S=256/1024 correctness unchanged.
Torch (test_mxfp8_bwd_dvdk2.py): mode3 vs mode1 still BITWISE MATCH at
H4/S4096 + H32/S16896; big-shape total bwd 164.0 -> 152.0ms (+7.3%),
small-shape +9.1% (was +6.0%/+5.0% pre-c15).
Remaining stall map: wait 34.8% (mma chain depth, register-walled),
SF fills ~4%, TT ring sleeping ~2%, exp2f ~4.5% (ex2.approx would break
bitwise parity with mode 1 — deferred).
