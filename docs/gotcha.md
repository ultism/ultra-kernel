# Gotchas — MXFP8 prefill attention for SM120a

Engineering notes for non-obvious traps found while building the kernel. Newest
first. Each entry: what bit us, the measured evidence, and the actionable rule.

---

## On sm120 `setmaxnreg` compiles to `USETMAXREG` (NOT `SETMAXNREG` — grep right) — the WS register realloc DOES work: producer R≤24, consumer R≤232. The spill wall is 232, not 168.

> **STATUS (2026-07-21): VERIFIED, correcting an earlier false alarm.** First pass grepped
> SASS for `SETMAXNREG`, found zero, and concluded "ptxas drops setmaxnreg on sm120" — WRONG:
> the sm120 SASS mnemonic is **`USETMAXREG`** (uniform datapath, no "N"), so the grep false-
> negatives. What actually happens (s3_kernel + minimal repro alike): PTX `setmaxnreg.dec 24`
> → `USETMAXREG.DEALLOC.CTAPOOL 0x18`; PTX `setmaxnreg.inc 232` → `USETMAXREG.TRY_ALLOC
> .CTAPOOL UP0, 0xe8` + a `BRA.U !UP0` retry loop (consumer spins until the CTA pool has the
> regs freed by the producer's dealloc). ptxas does PER-REGION allocation around the barrier:
> splitting the SASS at the two USETMAXREGs, the producer region tops out at **R20** (≤24) and
> the consumer at **R229** (≤232). `cuobjdump -res-usage`'s `REG:168` is only the LAUNCH-static
> per-thread count (65536/384 rounded down) — not the consumer's budget. **Consequence: the
> mainloop register wall is ~232, not 168.** The S9 pipeline experiments spilled against 232:
> full QK rotation with the accB form needs accO(64)+accB(64)+accS(32)+operands ≈ 220–240 →
> STACK 200B, net regression; the V-load hoist (tOrV live across the softmax) tipped past the
> margin too (STACK 232B). The accB fold (direct `accO += P*V`, −64 regs) funded the winning
> s9e/s9f shape. Live-range budget against **232** (and check `STACK` + region-split max-reg
> scans, never `-res-usage` alone) before ILP surgery on this kernel.

---

## The real-64 `(nb&1)` SF half-slice DEMOTES the SF fragments to a 160B stack frame — 1.5GB of byte-granular local traffic that the ncu "spill" wording hides. Fix: drive the n_block loop in even/odd PAIRS with `cute::Int<0/1>`.

> **STATUS (2026-07-21): FIXED (S9a, ~1.14–1.17× all configs).** `subSFK(tSrSFK, nb&1)` and
> `tOrSFV(_, _, (nb&1)*NKB + k)` index register fragments by a RUNTIME value → ptxas demotes
> them to local: `STACK:160`, SASS = 64× `LDL.U8` + 32× `STL.U8` per n_block (each QMMA's SF
> byte reloaded from stack), ncu local = 90% of L1TEX sectors at **0.3B/sector**, LSU the #1
> issue pipe (45.7% > tensor 42.1%). ptxas's "20B spill" counts only true spills — dynamic-
> indexed arrays live in the **stack frame**, not the spill count. The audit saw the traffic
> but called it "fire-and-forget, non-binding": wrong on the ISSUE axis (local reloads do hit
> L1 at 99%, but every LDL/STL.U8 steals an issue slot from QMMA/FFMA at 2 warps/scheduler).
> Fix: `for (nb; nb += 2) { step(nb, Int<0>); if (…) step(nb+1, Int<1>); }` with the half a
> `cute::Int<>` template constant → STACK 160→72B→48B, `LDL.U8/STL.U8` = 0, local sectors
> −98.4%, L2 SOL 76%→25%, LSU 45.7%→30.9%, tensor 42→51%. Same selection values, bit-exact.
> Rule: **NEVER index a CuTe register fragment by a runtime n_block parity** — make the parity
> a type-level constant by unrolling the loop 2× (or templating the body).

---

## The SF 128-granularity that blocks `kBlockN=64` is the sm100 **TMEM** staging format, REUSED on sm120 by inheritance — NOT an sm120 hardware requirement. The sm120 warp QMMA wants literally **1 scale byte per thread** (`SFARegisters = uint8_t[1]`); the interleaved 128-atom means nothing to it.

> **STATUS (2026-06-24): DONE + VALIDATED + COMMITTED** to branch `experiment/s8-real-n64` (commit
> `3bc93e0`), NOT merged to master. A genuine `kBlockN=64` DATA tile (not consumer sub-tiling) kills the
> consumer spill (ptxas 780B→**20B**) and runs **~1.4× faster than committed HEAD-128**
> (bench_ragged varlen+GQA+causal: 1.43/1.40/1.38× on (1,1)/(8,2)/(32,8)). ncu: `long_scoreboard`
> 2.76→**0.23** (= Sage 0.19), `issue_active` 29→**40%** (= Sage 42%). Mechanism below; the trick is the
> SF stays the cutlass 128-key atom (decoupled `kSFBlockN=128`), loaded per-64-block at coord `nbg/2`
> (host 128-pads kv) and **register-indexed to the `nb&1` 64-half** — no native SF rewrite needed. So the
> "change surface is large" warning at the end of this entry turned out **smaller than feared**: SF gmem
> layout untouched, only the consumer slice + host kv-pad + the data (not SF) box changed.

**TL;DR:** The whole "SF tile is 128 along its scaled dim, can't do a 64-key block" wall
(`Sm1xxBlockScaledConfig::Blk_MN = _128`) is **not** the hardware — it's the sm100 (B100/B200) **TMEM**
scale-factor layout (interleaved so `UTCCP` can stage SF into Tensor Memory for `tcgen05.mma`). sm120
(consumer Blackwell, RTX 50-series) has **no TMEM** and no `tcgen05`; its block-scaled MMA is the
**warp-level** `SM120_16x8x32_TN_VS`, whose SF is a plain **register operand**. We only carry the 128
atom because the cutlass sm120 collective `using`s the sm100 config verbatim. It costs us the spill: the
128 granularity forces `kBlockN=128` → PV operands stay 4 k-tiles = 16 regs (vs Sage nvfp4's 2 = 8) →
the consumer register spill that L2-bandwidth-bounds the kernel (entry below).

**The three source facts.**
- **The atom is the sm100 TMEM format** — `cutlass/detail/sm100_blockscaled_layout.hpp:48-59`:
  `Blk_MN = _128`; `SfKMajorAtom = Layout<Shape<Shape<_32,_4>,Shape<SFVecSize,_4>>, Stride<Stride<_16,_4>,
  Stride<_0,_1>>>`. The MN dim `(_32,_4)=128` has strides `(_16,_4)` = **interleaved** (a TMEM/UTCCP bank
  layout), so `tile_to_shape` makes a 512B indivisible atom → a 64-key box fails (`"CTA_Tile/SLayout
  top-level size equivalence"`). The K dim's `SFVecSize=32` has stride `_0` — *that* (the broadcast) is
  the only real hardware fact: one ue8m0 per 32-element block.
- **sm120 hardware wants 1 byte/thread, no interleave** — `cute/arch/mma_sm120.hpp:1941-1949`
  (`SM120_16x8x32_TN_VS<…,float_e4m3_t,float,float_ue8m0_t,VS>` and every sibling):
  `using SFARegisters = uint8_t[1];  using SFBRegisters = uint8_t[1];`. The `QMMA.SF` instruction takes
  exactly ONE scale byte per thread per operand (that thread's 32-key block). No 128, no interleave.
- **We inherited it** — `cutlass/gemm/collective/sm120_blockscaled_mma_tma.hpp:128`:
  `using Sm1xxBlkScaledConfig = cutlass::detail::Sm1xxBlockScaledConfig<SFVecSize>;` (and that file greps
  ZERO for `tcgen05`/`TMEM` — SF rides `gmem→TMA→smem→register`, never TMEM). Our `s3_kernel.cuh:160`
  `BlkSF = Sm1xxBlockScaledConfig<SFVecSize>` + `tile_atom_to_shape_SFA/SFB` ride it.

**Rule.** On sm120 the block-scaled SF gmem/smem layout is **ours to choose** — the only hardware
contract is "one ue8m0 per `SFVecSize`-block, delivered to each thread's `SFARegisters[1]` per the
m16n8k32 thread→scale map." Do NOT treat `Sm1xxBlockScaledConfig`'s 128 as immutable (it is a TMEM
artifact). **This supersedes the framing of the "head_dim < 128: SF atom is always 128" entry below**
(which correctly described the cutlass behavior but wrongly implied the 128 is a hardware floor — it's a
TMEM-format floor, droppable on sm120).

**What we actually shipped (the cheaper path that the SF-128 = TMEM insight unlocked).** We did **not**
rewrite SF to a native 64-layout — we kept the cutlass 128-key atom and just **half-sliced** it, which is
far less code and equally correct:
- The DATA tile is genuinely 64 (`kBlockN=64`) → PV operands `tOrP`/`tOrV` 16→**8 regs** = Sage level →
  spill gone (the whole point).
- The SF stays the 128 atom via a decoupled `constexpr int kSFBlockN = max(kBlockN,128) = 128`. The
  producer loads the full 128-SF box containing the 64-block (coord `sfg = nbg/2`; host 128-pads kv so the
  atom always aligns). The consumer then **register-indexes the correct 64-half**: QK uses a `subSFK`
  re-layout to pick the `nb&1` half of SFK; PV indexes `(nb&1)*NKB + k` into SFV. `kNCol = kBlockN/4`.
- **The half-slice is invisible to self-tests** (s6a dense ref = same kernel, so a swapped half stays
  bit-exact) — gate it on the **independent torchao oracle replayed at 64-key blocks** (`/tmp/test_real64.py`):
  a 128-block reference disagrees ~0.3% from P-quant granularity (NOT a bug), the 64-block reference gives
  ~**5e-6** = exact. This SF-half-swap trap is the one real hazard; everything else is mechanical.
- De-risked the SF mechanism on the cutlass GEMM example FIRST (`79c_..._mxfp8_..._gemm`): SFB free dim
  `N=64` tiles natively; the contraction/scale axis `K=64` would fail (`K/SFVecSize/Blk_SF=0`) but we never
  need K=64. 64-only, guarded by `static_assert(kBlockN==64)`; master stays 128.

---

## The dominant `long_scoreboard` is CONSUMER REGISTER SPILL, not K/V TMA latency — PROVEN by ncu-ing SageAttention3 at the IDENTICAL 128×128 tile (it barely spills). Root cause: mxfp8's k32 atom = 4 operand k-tiles vs nvfp4's k64 = 2, so we hold 2× the operand registers → over the 168 ceiling → `accO` spills. (Corrects the "latency/occupancy-bound" framing of the two entries below — that's the symptom, this is the cause.)

> **STATUS (2026-06-24): RESOLVED — the spill WAS the bottleneck; real `kBlockN=64` killed it (~1.4×).**
> The diagnosis below went through a correction arc worth remembering (see the "two false turns" box at
> the end of this entry): (1) the original "spill-bound, sub-64 is a +15-18% win" was **FALSE** — sub-64
> (consumer sub-tiling) is a net **regression** (1.37-1.52× slower) because it cut *static* spill 780→88B
> but **tripled dynamic local traffic** to ~3.3GB; (2) "spill is a red herring / pipeline hides it" was
> ALSO wrong — the kernel is **L2-bandwidth-bound and ~85% of L2 traffic is spill reloads** (local 1.66GB;
> consumer sm120's 99KB smem leaves ~1KB L1 → 1.5% local hit-rate → every spilled value round-trips L2).
> The real fix = a genuine 64 DATA tile (`accO` loop-carried, no save/restore churn), which dropped ptxas
> spill 780B→**20B** and won ~1.4×. See the top entry of this file + `[[s8-consumer-spill-bound]]`,
> `[[sf-128-atom-is-tmem-artifact-not-sm120-hw]]`.

**TL;DR:** A warp-specialized PERSISTENT kernel hides K/V latency via the async producer/consumer
pipeline at 1 CTA/SM **by design** — low occupancy is NOT the bottleneck (CUTLASS WS GEMMs run 1–2
CTA/SM at 80%+). Instruction-level profiling shows the dominant `long_scoreboard` (2.93/issue on
(8,2) causal varlen) is mostly the **consumer's OWN register-spill reloads** (`LDL` = load-local),
not K/V waits. The reference SageAttention3, at the *identical* tile/threads/regs, barely spills.

**Instruction-level evidence (`tests/bench_ragged.cu` (8,2) causal + ncu).**
- ptxas: `Used 168 registers` (the `__launch_bounds__(384,1)` ceiling = 65536/384), **~780B spill**.
  The runtime `setmaxnreg.inc<232>` does NOT remove what ptxas already baked against 168.
- ncu `--page source`: **`LDL [R1+0x17c]` is the #1 `long_scoreboard` instruction** (12.5% of all
  long_scoreboard directly; the dependent FFMA chain reloading the spilled `accO` adds more). The
  consumer reads K/V from smem = `LDS` = *short*_scoreboard, NOT long. Local-mem traffic **1.66 GB**.
- `-lineinfo` + nvdisasm: spill concentrated at **the PV/telescope** (`accO = accO*scale + accB`) —
  the two 64-fp32 accumulators (accO + accB) + the operands exceed 168.

**The decisive proof — ncu the SageAttention3 reference (prebuilt `.so`, runnable).** Sage3
head_dim=128 = `Flash_fwd_kernel_traits<128,128,128,3>` = the SAME 128×128 tile, 12 warps/384 thr,
SAME `launch_bounds(384,1)`, SAME `reg_dealloc<24>`/`reg_alloc<232>`, SAME accB-telescope. Causal:

| metric (ncu) | Sage3 (nvfp4) | ours (mxfp8, (8,2)) |
|---|---|---|
| `long_scoreboard` /issue | **0.19** | **2.93** |
| local spill ld / st | **9.88 MB / 0.85 MB** | **883 MB / 779 MB** |
| `issue_active` | **42.0%** | **29.3%** |
| registers / occupancy | 168 / 1 CTA | 168 / 1 CTA |

Sage essentially does NOT spill; its stalls are clean `short_scoreboard`+`wait` (a register-resident
WS attention). **42% issue / 0.19 long_scoreboard is the target if we kill the spill.** Run it:
prebuilt `tmp/SageAttention/sageattention3_blackwell/fp4attn_cuda.cpython-312*.so` — import torch
first, `sys.path.insert(0,'.')`, `from sageattn3.api import sageattn3_blackwell`, call on bf16
[B,H,L,D], `ncu -k regex:compute_attn_ws --launch-skip 1 --launch-count 1`. (A from-source device
compile FAILS on CUDA 13.3 — its bundled cutlass `prefetch.hpp` + conda-gcc `fenv.h` conflict; use
the prebuilt `.so`.)

**Root cause = operand k-tile COUNT, not per-MMA width.** Per-MMA operand width is IDENTICAL (both
4 regs: mxfp8 `SM120_16x8x32` A = m16k32 e4m3 = 16B; nvfp4 `SM120_16x8x64` A = m16k64 e2m1 = 16B).
But head_dim 128: mxfp8 k32 → **4 k-tiles**, nvfp4 k64 → **2**. cute holds the full operand fragment,
so `tSrQ`/`tOrP`/`tSrK`/`tOrV` = **16 regs each vs Sage's 8**. Peak: Sage accO 64 + accB 64 + ops ~16
≈ 145 (fits 168); us 64 + 64 + ops ~32 + temps ≈ 175 (spills). The accumulators are identical for both;
the operand k-tiles are the whole gap. (nvfp4 vs mxfp8 = a format choice we can't make — mxfp8 is the
project.)

**Two fixes tried this session — both DEAD (gated, reverted):**
- **accB elimination / in-place PV** (`S3_INPLACE_PV`): rescale `accO` then `gemm(...,accO)` directly
  (FA2-style, no per-block `accB`). **4.7× SLOWER** (1.14→5.30 ms), `long_scoreboard` 2.93→4.33, spill
  784→920 B. `cute::gemm` PINS its C accumulator in registers for the whole gemm — a short-lived
  per-block `accB` lets ptxas spill the long-lived `accO` COLD/flexibly; making `accO` the MMA target
  pins all 64 of it across every block → operands spill HOT. **`accB` is a load-bearing escape valve,
  NOT redundant** — Sage uses the SAME accB-telescope, confirming the baseline.
- **operand streaming** (`S3_STREAM_V`, single-k-tile reused fragment): **cute structural wall.**
  `retile_D` (the smem→rmem copy/MMA layout bridge) is built for the FULL operand fragment; a
  single-k-tile fragment fails to compile (`logical_divide: too many modes in tiler`). Sage's own
  `copy_v_block` retiles the FULL fragment then slices — it does NOT reduce registers either. "Stream
  to fewer *held* registers" is not a clean cute pattern; nobody (incl Sage) does it.
- knobs tapped: reg_alloc 232→240 only −6% spill (784→736 B); dropping `minBlocks=1` → WORSE (→1180 B).

**The clean lever — WON: `kBlockN` 128→64 (real DATA tile, NOT consumer sub-tiling).** PV k-tiles 4→2
(= Sage), `accS` 64→32, `tOrP`/`tOrV` 16→8; `accO`/`accB` unchanged → PV peak ~176 → ~144 < 168 →
**spill dropped ptxas 780B→20B, ~1.4× faster than HEAD-128** (long_scoreboard 2.76→0.23, issue 29→40% =
Sage level). The tradeoff prediction held: 2× n_blocks = 2× K/V stream, but K/V rides the **hidden** TMA
pipeline (short_scoreboard) while spill `LDL` is **unhidden** (long_scoreboard) — trading unhidden spill
for TMA-hidden K/V netted the win. It touched exactly the predicted surface (SF-keys layout / Blk_SF-128
trap, the S5 P-shuffle's per-keys SF blocks, masks/kFillZero/TMA tiling, and the `n_block_max`
producer/consumer contract), resolved per the top entry of this file (SF kept at the 128 atom + half-sliced).
Committed to branch `experiment/s8-real-n64` (`3bc93e0`), 64-only via `static_assert`, NOT merged.

**TWO FALSE TURNS before the win (don't re-walk them).**
1. **`sub-64` (consumer sub-tiling) is a CHURN-TRAP, not a fix.** Faking a 64-block by no-unroll
   register-reuse inside a 128-loop cut ptxas *static* spill 780→88B but **raised dynamic local traffic
   to ~3.3GB** (operand re-reads + accO save/restore per sub-block) → net **1.37-1.52× SLOWER** than
   HEAD-128. Static ptxas spill bytes ≠ dynamic local traffic (= L2 bandwidth). A genuine 64 tile keeps
   `accO` loop-carried (no save/restore) and avoids the churn entirely.
2. **"spill is a red herring, the persistent pipeline hides it" was ALSO wrong.** It LOOKED true:
   the committed HEAD-128 was the fastest build yet had long_scoreboard 2.76 (≈ the "2.93 bottleneck"),
   and `issue_active%` predicted wall-clock better than long_sb. But the deeper probe showed the kernel
   is **L2-BANDWIDTH-bound (SOL L2 86%) and ~85% of that L2 traffic IS spill** (local_ld 883MB +
   local_st 779MB = 1.66GB; global LSU ~0 because K/V are tiny fp8 + L2-resident; local **L1 hit-rate
   1.5%** because 99KB smem leaves ~1KB L1). So the spill's *latency* is hidden but its *bandwidth*
   saturates L2 — the original spill instinct was right, the *mechanism* was BW not latency. Killing the
   spill at the root (real-64) is what cashed it in.

**Rule.** For a WS persistent kernel, "low occupancy → latency-bound" is the WRONG diagnosis — the
pipeline hides K/V latency at 1 CTA/SM by design. Profile `long_scoreboard` at **instruction level**
(`ncu --page source --print-source sass --metrics ...long_scoreboard`): if `LDL`/`STL` dominate it is
**register spill**. But two follow-on traps: (a) `long_scoreboard` ANTI-correlates with wall-clock here
(the pipeline hides the spill *latency*) — judge by `issue_active%` + the **L2/local-traffic SOL**, not
long_sb; the spill bites as **L2 bandwidth** (1.5% L1 hit → every reload round-trips L2), not stall
latency. (b) Cut the spill at the ROOT (fewer live registers via a genuine smaller tile / fewer operand
k-tiles), NOT by restructuring to a smaller *static* spill number — `sub-64` did the latter and tripled
dynamic local traffic. The two entries below describe the *symptom* (latency/occupancy at 1 CTA/SM);
this spill is the *cause*. NOT a blocker even before the fix: we beat fa2 2–2.5× WITH the spill (it was
headroom, SM ~37%); real-64 then converted the headroom (~1.4× on top).

## GQA K/V reuse (the "fewer loads" lever) is UNREACHABLE on our 1-CTA/SM design: folding 2 qo_heads into one CTA to share K/V needs a 2nd O accumulator, which SPILLS — and the spill traffic is bigger than the K/V it saved (4.8× slower, `long_scoreboard` 9.4× WORSE)

**TL;DR:** The previous entry guessed GQA K/V reuse ("one CTA serves a kv_head's whole qo_head group")
might be the lever. We built it as a gated probe (`S3_GQA_FOLD`) and **measured it: catastrophic.**
The reuse direction is real (fa2 does it) but it is **not affordable on our register-bound design** —
holding *k* heads' online-softmax O accumulators to share one K/V stream needs *k×* the persistent
register state, and we are already at the spill floor at 1 CTA/SM.

**The probe.** One CTA processes 2 qo_heads of one kv_head (config (8,4), group=2): K/V TMA-loaded
ONCE per `n_block`, an inner `for h` loop does both heads' QK→softmax→PV into **two independent
`accO` accumulators** (the dead S5-path `sP`/`sSFP` smem reused as head-1's `sQ2`/`sSFQ2`, so smem
didn't grow). Halves the K/V load count vs the per-`qo_head` baseline. Clean A/B, same shape:

| metric (ncu, clock-independent) | baseline (per-qo_head) | FOLD (2 heads/CTA) | |
|---|---|---|---|
| wall-clock (8,4) causal | 1.22 ms | **5.86 ms** | **4.8× slower** |
| `long_scoreboard` /issue | 2.79 | **26.11** | **9.4× WORSE** |
| local-mem **store** (spill) | 779 MB | **2.60 GB** | 3.3× |
| local-mem **load** (spill) | 883 MB | **2.74 GB** | 3.1× |
| SM throughput | 39.0% | **15.5%** | halved |
| occupancy (warps active) | 18.7% | 18.7% | still 1 CTA/SM |

**Why it backfires.** A 2nd `accO` (64 fp32/thread) pushes the consumer's live registers far past the
**168-reg static ceiling ptxas allocates under `launch_bounds(384,1)`** (the consumer's runtime
`setmaxnreg.inc<232>` can't change what ptxas already spilled against 168). Result: **2.6 GB of local
stores** — *more* memory traffic than the K/V loads the reuse was trying to remove. Worse, spill ld/st
ride the **same long-latency scoreboard** as the K/V loads, so the metric we were trying to lower
(`long_scoreboard`) went UP 9.4×. The arithmetic is unforgiving: any scheme that processes >128
accumulator-rows per K/V stream (the *only* way to cut the load count — head-fold into a fixed 128-row
tile does NOT, it keeps 128 rows so the loads are identical) needs >128 rows of live `accO`, which our
256-thread/128-row/232-reg layout cannot hold without this spill.

**Rule.** On a kernel pinned at 1 CTA/SM and already at the register spill floor, "issue fewer loads
via K/V reuse" is a **trap**: the extra accumulator state spills, and spill traffic + spill-scoreboard
stalls cost more than the loads saved. Capturing GQA reuse requires fa2's structure — **smaller tiles
/ fewer threads** so *k* heads' accumulators fit without spilling — i.e. a ground-up tile/thread
redesign, not a tuning knob. Don't confuse "one CTA covers the group" (free) with "fewer K/V loads"
(needs >128-row accumulators = more registers). Probe reverted; reproduce by re-adding the gated
2-head inner loop. Corrects the prior entry's parenthetical hope and `[[s8-tuning-latency-bound-depth-exhausted]]`.

## Deeper TMA pipelines (K `kStages`, V double-buffer) do NOT help a kernel that is latency/occupancy-bound at 1 CTA/SM — and V double-buffering REGRESSED it. More in-flight loads with no spare warps to overlap just adds memory contention

**TL;DR:** On production varlen+causal+GQA shapes our kernel is **memory-latency-bound at 1 CTA/SM**
(`long_scoreboard` ≈ 2.78 stalls/issue dominant; DRAM only ~40% so NOT bandwidth-bound; tensor < 39%
so NOT compute-bound; L2 hit 94.6% so the GQA per-`qo_head` K/V reloads are absorbed; achieved
occupancy 18.6% = **1 CTA/SM**, register- AND smem-locked). The textbook fix for a latency stall is
a deeper TMA prefetch pipeline. We tried both rings; **the clock-independent metric refused both**:

| experiment | `long_scoreboard` | verdict |
|---|---|---|
| baseline (K `kStages=2`, V depth-1) | 2.78 | — |
| K `kStages` 2→3 (after sP-reclaim freed the smem) | 2.76 | flat — no help |
| **V depth 1→2** (mirror the K ring) | **3.27** ↑ + DRAM↓ + SM-throughput↓ | **REGRESSED** |

**Why depth can't help here.** A deeper ring hides latency only if, while a warp waits on its load,
**other resident warps have independent work to issue**. At 1 CTA/SM with the consumer warpgroup
register-locked (`setmaxnreg.inc<232>`, tuned to the spill floor) and smem pinning a single CTA,
there is **no second CTA and no spare warp** to overlap into the wait. Adding a 2nd V buffer just
puts a 2nd V TMA in flight concurrently with the 2 K TMAs → more outstanding requests contend for
the same DRAM/L2 path → each load's *effective* latency grows → `long_scoreboard` goes **up**. The
kernel got slower (DRAM% and SM-throughput% both dropped because wall-time rose), while staying
**bit-exact** (s6a_ragged O max|abs|=0, torchao 12/12) — a pure perf regression, easy to misread as
"working" if you only check correctness.

**The wall-clock was useless here — trust ncu.** Home-clock wall-time for the same runs swung the
(32,8) config **+32%** between builds (thermal/clock noise on a ~5 ms kernel); the (1,1) showed a
spurious −7%. Only `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio`
(per-issue, clock-independent) gave a stable read. Profile one steady launch with
`ncu -k regex:s3_kernel --launch-skip N --launch-count 1 --metrics ...long_scoreboard...`.

**Rule.** Pipeline depth is a lever **only when occupancy leaves spare warps to overlap**. For a
kernel already pinned at 1 CTA/SM, do NOT reach for `kStages`/double-buffering — it's a no-op at best
and net-negative at worst. The real levers for such a kernel are (a) **raise occupancy** (more
CTAs/warps — here blocked by registers+smem, and the 2-CTA-via-smaller-tiles route was ruled out by
design) or (b) **issue fewer / cheaper loads** (e.g. GQA K/V reuse: one CTA serves a kv_head's whole
qo_head group). And measure depth changes with ncu `long_scoreboard`, never the home wall-clock.

## Making a TMA load conditional means shrinking `transaction_bytes` in lockstep — the arrival barrier waits on a fixed byte count, so a load you drop without dropping its bytes DEADLOCKS (same hang class as the n_block_max mismatch)

**TL;DR:** S6b's per-tensor-fp8 SFSource (`kUniformFp8`) consumes a mainstream `torch.float8_e4m3fn`
KV cache that carries **no** scale factors: the kernel synthesizes a uniform byte-127 (`2^0 = 1.0`)
block scale in registers and **skips every SF TMA load**. But a `PipelineTmaAsync` arrival barrier
is armed with `transaction_bytes` = the EXACT byte count the producer will deposit; the consumer's
`consumer_wait` unblocks only once the mbarrier has counted that many bytes arrive. Drop the SF
`copy(params.tma_sf*…)` *without* subtracting its bytes from `transaction_bytes` and the barrier
waits forever for bytes that never land → producer and consumer both block → **100% GPU hang, no
output** (NOT a wrong number) — the SAME failure class as the [n_block_max producer/consumer
contract](#a-causal-mask-change-in-a-warp-specialized-kernel-is-two-edits) below.

**The contract (a PAIRED edit).** Skip-the-load and shrink-the-byte-count live together. In
`s3_kernel.cuh`, both are keyed off the same `kLoadSF = (Src == SFSource::kMxFp8)`: (1) the producer
wraps each SF `copy()` in `if constexpr (kLoadSF)`; (2) the pipeline params set
`transaction_bytes = kLoadSF ? TmaBytes* : TmaData*` (data-only). `TmaBytes*` was split into
`TmaData* + TmaSFBytes*` precisely so the data-only count is exact. Q/K/V each TMA BOTH their data
**and** their SF under ONE barrier, so each barrier's byte count must drop by exactly that operand's
SF contribution — no more, no less. (The consumer-side `if constexpr` that fills the const-127 SF
fragment in registers is harmless on its own; it's the *producer load + the byte count* that form
the deadlock-able contract. Marked `[SF-bytes contract]` at all three sites — grep it.)

**Rule.** Any conditional TMA load sitting behind a single arrival barrier must adjust that
barrier's expected-bytes in the same breath. If a WS/TMA kernel hangs right after you made a load
optional, suspect `transaction_bytes` before anything else. (This generalizes the n_block_max
lesson: a *trip-count* mismatch and a *byte-count* mismatch are the two ways to starve the same
producer/consumer pipeline into a silent hang.)

---

## A causal-mask change in a warp-specialized kernel is TWO edits — the producer's `n_block_max` (TMA load bound) and the consumer's must stay byte-identical, or the pipeline DEADLOCKS (100% GPU hang, not a wrong number)

**TL;DR:** Adding the slice-3 append/chunked-causal offset (`offset_q = kv_len - qo_len`, so query
`m` attends keys `[0, m + offset_q]`) means changing **two** things: the masking predicate
(`n_local > m_local + offset_q`) **and** the `n_block_max` causal bound
(`ceil((q_tile+1)*kBlockM + offset_q, kBlockN)`). The bound appears in **both** warpgroups — the
**consumer** uses it to size its QK/softmax/PV loop, and the **producer** (TMA-load warp) uses the
*same* `n_block_max` to decide how many K/V tiles to stream into the pipeline. I edited only the
consumer's copy first. For `offset_q > 0` the consumer then `consumer_wait`s on more K-blocks than
the producer ever `producer_acquire`/loads → the pipeline starves and **both warpgroups block
forever**. Symptom: the kernel **hangs at 100% GPU utilization** (no output, no error, no NaN) —
qualitatively different from a masking bug, which returns a *wrong* finite number.

**How it bit us.** After changing only the consumer `n_block_max` (`s3_kernel.cuh` consumer block),
`tests/s6a_ragged.cu` with the new `kv_len > qo_len` requests hung; `nvidia-smi` showed the test
process pinned at 100% util with an empty (block-buffered, never-flushed) stdout. The fix was to
apply the *identical* `offset_q` expression to the **producer's** `n_block_max` (`s3_kernel.cuh`
producer block — which also had to start reading `qo_len = get<5>(bc)`, previously unused there).
After matching them: ragged bit-exact `O/LSE max|abs| == 0` at 1/2/36 SM, torchao oracle matches.

**Rule.** In a WS attention kernel, `n_block_max` (and any per-tile loop bound the producer and
consumer share) is a **contract** between the two warpgroups. Any change to it is a paired edit —
grep for *every* site that computes it and change them together. A divergence doesn't corrupt
output, it **hangs**; if a WS kernel pins the GPU with no progress after a loop-bound change,
suspect a producer/consumer trip-count mismatch first. (Belt-and-suspenders: the dense
single-tile path has `offset_q == 0`, so this only ever manifests once `kv_len > qo_len` requests
exist — i.e. it stayed invisible through slices 1–2 and first appeared in slice-3.)

---

## Block-scale PV: a masked key with `P==0` still poisons `accO` if its padded V is NaN — the hardware mxf8 MMA computes `0 * NaN = NaN`, and the QK mask does NOT protect the V side

**TL;DR:** When `kv_len` is not a multiple of `kBlockN`, the last K/V tile is **partial**:
the padded keys `[kv_len, kBlockN)` are loaded into smem as whatever the buffer holds. The
QK mask sets their softmax weight `P = 0` **exactly** (`accS = -inf → exp2 = 0 → e4m3 0x00`),
so on the QK side they vanish. But **PV reads V (data **and** SF) for ALL keys including the
masked ones**, and the SM120 block-scaled MMA `SM120_16x8x32_TN_VS` propagates `0 * NaN = NaN`
(it is **not** absorbed to 0). A padded **V-DATA** byte that is e4m3 NaN (`0x7F`/`0xFF`) **or**
a padded **V-SF** byte that is ue8m0 NaN (`0xFF`) therefore turns `0 * NaN` into `NaN`, which
poisons the **entire** `accO` row (every head_dim column) → the whole query row's `O` is NaN.
This is the **K/V asymmetry**: K is masked to `-inf` *before* softmax so its garbage (even NaN)
never reaches an accumulator; **V is read *after*, and `P==0` is not a shield against NaN.**

**How it bit us (and the proof).** Partial-len requests (`kv_len` 200/360) with the pad
gap poisoned `0xFF` (`tests/s6a_ragged.cu`): **no-fix → `ragged=nan`, `bad=71680`** on exactly
those requests; with the fix → `O max|abs| == 0` bit-exact. This is the SAME failure FlashInfer's
NVFP4 prefill guards against with a `kFillZero` V-SF load (`prefill.cuh:531-532`: *"0 (softmax
weight) * NaN (uninitialized SF) = NaN"*) — except theirs is **software** (`__hmul2` the SF into
an upcast-bf16 V, strict-IEEE in CUDA cores), while ours is the **hardware** block-scaled MMA and
it bites identically. (Note: FlashInfer has **no MXFP8 block-scale attention** at all — only
NVFP4; MXFP8 block-scale lives only in their GEMM templates.)

**Why it's a footgun.** Our design 128-pads **per request** (descriptor extent = padded len),
so TMA never sees the partial block as OOB → no free TMA OOB-zero (unlike FlashInfer's *tight*-
packed ragged, where an interior partial block reads the **next** request's FINITE data and only
the buffer tail is hardware-zeroed). The gap between requests is genuinely uninitialized. A real
recycled/paged KV-cache tail is exactly this. And `P==0` *looks* like it should make the term
vanish — it doesn't.

**Rule — kFillZero the partial block's V before PV (gated `S3_V_KFILLZERO=1`).** TMA can't
predicate a load per-row, so sanitize *after* the load, only on `partial_n`: (1) zero the padded
V-DATA columns `[valid, kBlockN)` in `sV` smem (`as_position_independent_swizzle_tensor(sV)(hd,key)
= Element(0)` + a `NamedBarrier` before the ldmatrix) — `valid = kv_len - nb*kBlockN`; this also
covers the **straddling** 32-block's masked keys (whose SF is real/finite, shared with its valid
keys). (2) Finite-ize the **register** SF fragment `tOrSFV(_,_,k)` for **fully-masked** 32-key
tiles (`k*SFVecSize >= valid`) — the straddling tile keeps its real SF. **Zero BOTH data and SF:**
zeroing only data leaves `0 * 2^(NaN_SF)`; zeroing only SF leaves `0 * NaN_data`. (SF is per-32-
block so you can't zero just the masked keys of a straddling block without corrupting its valid
keys — hence data-zero handles those, SF-finite-ize handles only fully-masked blocks.)

---

## A `abs(a-b) > tol` differential comparator is BLIND to NaN — `NaN > tol` is `false`, so a NaN kernel output silently certifies as PASS

**TL;DR:** In any numeric diff test, `e = std::abs(a - b); if (e > tol) ++bad;` **cannot detect
a NaN** result: IEEE makes every ordered comparison with NaN `false`, so `NaN > tol` is `false`
(not a failure) and `NaN > max_abs` is `false` (so even `max_abs` stays clean). A kernel emitting
all-NaN sails through as `bad=0, max_abs=0` → **false PASS**.

**How it bit us.** The adversarial `0xFF`-pad test (above) FALSELY PASSED both **with and without**
the `kFillZero` fix. The tell: poisoning even a **valid** key (which has `P≠0`, so a real NaN must
propagate) *still* "passed" — impossible unless the comparator was eating NaN. Rewriting to
`if (!std::isfinite(a) || !(e <= tol)) ++bad;` (note `!(e<=tol)`, which is **true** for NaN, vs
`e>tol` which is false) immediately exposed `ragged=nan, bad=71680`.

**Rule.** Any differential/oracle test whose kernel *can* produce NaN/Inf (block-scale, softmax,
division by a row-sum, masked rows) MUST check `isfinite` explicitly — `> tol` alone silently
certifies NaN as correct. Cousin of the self-replay-vs-independent-oracle lesson (a self-consistent
test can be blind to a whole bug class); here it was the **comparator itself** that was blind.

---

## TMA: the shape you pass `get_tma_tensor` is only a *nominal* extent — indexing a tile past it does NOT error (arithmetic coords, no bounds check); the host descriptor does the real addressing

**TL;DR:** A CuTe SM90 TMA copy has **two** "tensors". (1) The **descriptor**
(`CUtensorMap`), baked **host-side** by `make_tma_copy(op, gmem_tensor, …)` from the
gmem tensor's real base address + global extents + strides + box shape — this is what
actually generates addresses and does OOB predication, and it is fixed at construction.
(2) The **coordinate tensor** returned by `get_tma_tensor(shape)` — its "data" is not
memory but a stream of *coordinates* (engine = `ArithmeticTupleIterator`, basis strides
`E<0>,E<1>,…`); it only feeds CuTe's layout algebra (`local_tile`/`partition_S`) to decide
which tile/thread lands at which coordinate. **The `shape` you pass it is purely nominal:
it sets the coordinate tensor's stated extents (and thus the nominal tile count), but
`crd2idx` is plain arithmetic `coord·stride` with NO bounds check, so indexing a tile
index *beyond* that nominal shape silently produces the correct out-of-nominal coordinate,
which the descriptor (built over the full/packed tensor) then maps correctly.**

**How it bit us (and the proof).** Pre-cleanup our K-SF did
`mSFK = tma_sfk.get_tma_tensor(shape(layout_sf))` where `layout_sf` was the **per-128-block**
`(128,128)` SF layout — nominal key-block count **1** — yet the producer indexed
`tKgSFK(_, nb)` for `nb = 0..3` and loaded all four K-SF blocks **correctly**. A probe
(`tests/sf_tma_probe.cu`, since removed) confirmed: `local_tile` reports the n-block mode
`size = 1`, but `gSFK(key=0, hd=0, nb)` for `nb=0,1,2,3` returns block-coords `0,1,2,3` —
no error, no clamp. The descriptor (built host-side over the **full** `SFA(SK,…)`) does the
addressing; the nominal `(128,128)` shape only ever governed the (ignored) bound.

**Why it's a footgun, not a feature.** Because TMA has **no bounds check anywhere**, a
mis-address (wrong base, wrong tile index, wrong head/batch stride) does **not** crash and
does **not** error — it silently reads the wrong gmem region (or hardware-OOB → zero-fill).
The *only* thing that catches it is a **bit-exact numerical** differential test
(our ragged/GQA tests assert `O max|abs| == 0` vs a per-request/per-head dense oracle).

**Rule — don't rely on out-of-nominal indexing in production code, even though it works.**
Every FlashInfer / CUTLASS TMA kernel passes the **FULL** global shape to `get_tma_tensor`
for data *and* SF (`hopper/mainloop.cuh:166` `layout_Q.shape()`; `blackwell/prefill/
mainloop_tma_ws.h:468` `shape(layout_SFK)` over full seqlen; `sm100_fmha_*`
`make_shape(H,D,B)`), and expresses batch/ragged/paged by **offsetting the coordinate
within** that full shape — they never index past nominal. The CuTe no-bounds-check property
is universal but they keep `nominal ⊇ index` **by discipline** (readability + intent: the
shape documents the real extent). We aligned to this (commit `05a8ced`): the kernel now
builds the **full** SFK nominal shape in-kernel from `seqlen_k` (`+ num_kv_heads` for GQA)
via the `CUTE_HOST_DEVICE` `tile_atom_to_shape_SFA`, so the n-block / head index always
stays within nominal. Addressing is byte-identical (the host descriptor never changed) ⇒
dense + ragged stayed bit-exact + torchao 4/4. The vestigial `params.layout_sf` (per-block)
field was removed. **Takeaway:** treat the `get_tma_tensor` shape as a *contract that should
equal the real extent*; matching it costs nothing and removes a silent-mis-address trap that
a reviewer would (rightly) flag.

---

## S5: filling one MMA operand from another's accumulator by `__shfl` — work in `recast<uint32>` words and prove the source-word index is warp-uniform

**TL;DR:** To hand P from the QK accumulator (PV is back-to-back with QK) straight into
the PV-A register operand without an smem transpose, you shuffle e4m3 bytes across the
quad. The instinct is a per-element gather (each dest e4m3 ← some lane's e4m3), but
`__shfl_sync` moves 32-bit words and **can't index a remote lane's register array by a
runtime index** — the source lane doesn't know which of its values you want. The clean
formulation: pack each thread's quantized bytes into `uint32 qw[...]`, and for each
**destination** `recast<uint32>(tOrP)` word, issue ONE `__shfl_sync(mask, qw[g], srcLane)`
per source lane with a **per-lane `srcLane`** but a **compile-time-uniform word index `g`**.

**Why uniformity matters:** `__shfl_sync` reads the *same named variable* from the source
lane, so `g` must be identical across all lanes that co-issue the instruction, while
`srcLane` may differ per lane. For our mxfp8 m16n8k32 maps (QK-C key→lane `(key/2)%4`,
PV-A key→lane `(key/4)%4`), the dest word for `(q-row r, e2, mk)` needs source word
`g = e2 + 2*mk`, and `g` is warp-uniform precisely because the dest key base `Kd = 4*L +
16*e2 + 32*mk` has `4*L < 16`, so the lane term drops out of `g = Kd/16`. That single
fact is what makes the whole thing one `__shfl` per word instead of a scatter. The two
source lanes are `{qb + 2*(L&1), qb + 2*(L&1) + 1}` (`qb = lane & ~3`, `L = lane%4`), and
`half = (L>>1)&1` selects the low-16b (L∈{0,1}) vs high-16b (L∈{2,3}) of each fetched word.

**Rule:** when redistributing an MMA operand across lanes, (1) cast both sides to the
byte-packed `uint32` view and reason per-word, not per-element; (2) derive the source-word
index symbolically and **verify it's uniform across the warp** before writing the `__shfl`
— if it isn't, restructure the packing until it is; (3) get the constants from a layout
probe (`partition_C` vs `partition_A`/`partition_B` over an identity tensor), never by hand.
Validate bit-exact against the smem-transpose path kept behind a macro, and on RANDOM data
(constant data is layout-blind — see the V-SF entry). Measured payoff: −11.7%/−11.8%
cycles (ncu base-clock) from deleting 2 NamedBarriers + a 16KB smem round-trip per n_block.

---

## Measuring O error: normalize by max|O|, never per-element `clamp_min` (it explodes on near-zero O)

**TL;DR:** Attention output O has entries that are legitimately ~0 (rows where the
softmax-weighted V sum cancels). A per-element "relative error"
`(O - O_ref).abs() / O_ref.abs().clamp_min(1e-3)` divides a real ~0.05 absolute
quant error by the `1e-3` floor at those near-zero entries → a **bogus 49.5 ("4950%")**
that looks like catastrophic failure when the kernel is actually fine. We hit exactly
this comparing fixed-vs-dynamic P quant: the headline number was `49.5`, the honest
number was `~1%`.

**The trap:** `clamp_min(floor)` turns "small error at a small-and-unimportant O
entry" into a huge ratio, because the denominator stops shrinking but the numerator
doesn't. The near-zero O entries it amplifies are the ones that **don't matter
downstream** (they're dominated by accumulation noise and feed a near-zero
contribution to the next layer).

**Rule:** report the max-abs O error **normalized by the global scale `max|O|`**
(`(O-O_ref).abs().max() / O_ref.abs().max()`), or just the raw abs error alongside
the O range. That gives the true "fraction of the signal" error (~1%), and it's the
metric that actually predicts downstream impact. Reserve per-element relative error
for quantities with no near-zero entries (e.g. row_sum, LSE). Same reasoning is why
the kernel's own asserts use `assert_close(rtol, atol)` (atol catches the near-zero
entries) rather than a pure rtol.

---

## Softmax P: a fixed scale 256 beats per-block dynamic MX (dynamic MX saturates the top 1/8 of each binade)

**TL;DR:** The PV A-operand is the softmax output P. Post-max-subtraction `p =
exp2((S - m_cur)*sm)` with `m_cur` the running max ⇒ **every P ≤ 1.0 and the row
argmax = exactly 1.0**. So quantize P with a **fixed scalar scale = 256.0**
(`se = -8`, `P_e4m3 = P*256`) — drop the per-32-block `amax`/`quad_reduce` and the
SF smem store/gather entirely (`tOrSFP` = the compile-time constant byte `-8+127 =
119`). `1.0*256 = 256 ≤ 448` (e4m3 max) ⇒ **never saturates**.

**The surprise — fixed is MORE accurate than dynamic, not just cheaper.** Measured
(`test_attn_matches_torchao`, max-abs O error vs un-requantized P, normalized by
max|O|): fixed `0.55–1.08%` vs dynamic per-block `0.86–2.56%` across hd64/128 ×
causal. Dynamic OCP-MX picks `se = floor(log2(amax)) - 8`, mapping amax into
`[256, 512)` — but **e4m3 max = 448 = 1.75·256, not 512**. Any block whose amax is
in the top 1/8 of its binade (e.g. amax=0.9 → 0.9·512 = 460.8) **saturates to 448**,
clipping the *largest* (most important) P values. Fixed-256 puts the max at exactly
256 and never clips; it only loses the tiny elements of small-amax blocks (below
~2^-17 → 0), which contribute <0.01% to `O = ΣP·V/ΣP`. For any quantity bounded by
1, a static scale dodges MX's systematic top-of-binade saturation.

**Rule:** for softmax P (or any [0,1]-bounded operand) use a fixed scale, not MX
dynamic. Toggle `S3_P_DYNAMIC_SCALE` keeps the dynamic path for A/B only. (When
reporting the precision delta, normalize by `max|O|` — see the measurement-trap
entry above; per-element relative error gave a bogus `49.5` vs the honest ~1%.)

**Measured — but trust ncu, not wall-clock, on a 0.1 ms kernel.** `bench/bench_p_quant.cu`,
RTX 5060 Ti. The clock swings 1537↔2820 MHz between runs (timing sd up to 0.27 ms),
so a min-of-200 read 10%/23% with the causal number inflated by clock noise. The
honest evidence is **ncu `--clock-control base`**, deterministic `sm__cycles_elapsed`
(4096²): non-causal **−10%** (291.6k → 262.0k cycles), causal **−17%** (310.9k →
256.9k); instructions −15%/−12%; **occupancy identical** (18.7%, 168 reg, 1 CTA/SM) so
it's not an occupancy effect. Mechanism: the `amax` reduction is a warp `SHFL`
(`quad_reduce`) whose result is consumed immediately → a `short_scoreboard` stall
(0.53→0.32) that, at this kernel's low ~18.7% occupancy, isn't hidden and lands on the
QK→PV critical path. A 3rd build (`-DS3_P_CONST_SF=0`, fixed scale but SF still via
smem) shows the whole win is the amax removal — dropping the SF smem store/gather adds
~0. Two takeaways: (1) on sub-ms kernels with boosting clocks, profile cycles with ncu,
don't trust wall-clock min; (2) bench needs the debug dumps off — guard the full-P gmem
write with `if (params.out_Ppre != nullptr)` or it dominates the time.

---

## head_dim < 128: the block-scaled SF smem atom / TMA box is always 128 (pad it, don't shrink it)

**TL;DR:** The cutlass block-scaled SF tile is inherently **128 along its tiled
dim** (`Blk_MN=128`, `Blk_SF=4` ue8m0 per 128-element K-block). Hand-rolling the SF
smem atom with the real `head_dim=64` makes `sSF_shapeK = K/SFVecSize/Blk_SF =
64/32/4 = 0` → a malformed layout → the TMA fires `"CTA_Tile and SLayout top-level
size equivalence"` / `"Could not find a common tile-gmem vectorization"`
static_asserts. There is **no sub-128 SF atom** in cutlass — normal GEMMs always
tile K to 128, and the collective only special-cases *N* < 128 (`logical_divide`),
never *K* < 128.

**The fix is to PAD, not shrink.** `tile_atom_to_shape_SFA(make_shape(128,128,64))`
already returns the **128-K-padded** layout (cosize 512, byte-identical type to the
K=128 case); `tile_atom_to_shape_SFB` pads N<128 the same way. So use
`kSFPadHD = max(kHeadDim,128)` (=128) for the **SF smem atom, `LayoutSF`, and all
three TMA SF boxes** (SFQ/SFK K-dim, SFV N-dim), while the **data path**
(`SmemLayoutQ/K/Vt`, data TMA boxes, the QK `TiledMma` Tile) keeps the real
`kHeadDim`. The QK MMA contracts only 64 (`size<2>(tSrQ)=2`), so the gemm loop reads
the first 2 of the 4 SF blocks; the host fills `NBLK=kHeadDim/32` real blocks and
cutlass auto-pads the gmem buffer (the upper blocks are loaded-but-unused). The V-SF
box becomes (128,128) and the same `SmemLayoutSFV` serves both head dims.

**Trap:** the TMA-SF box is built in THREE files that must agree on the Params type —
`tests/s3_kernel.cuh` (the `TMA_SF*` aliases), `tests/s3_e2e.cu` main, and
`tests/csrc/mxfp8_attn_ext.cu`. Updating only two of three compiles fine at
head_dim=128 (everything is 128 anyway) and fails **only** at head_dim=64 with a TMA
type mismatch — and only in the torch-extension TU, not the plain-nvcc self-test.

---

## FMA contraction in the online softmax → max P = 0.99999988 → e4m3 saturates the max to 7/8

**TL;DR:** Writing the softmax as `p = exp2f(accS*sm_log2 - max_scaled)` with
`max_scaled = m_cur*sm_log2` lets nvcc contract it to `fma(accS, sm_log2, -max_scaled)`.
At the argmax `accS == m_cur`, but the FMA subtracts the **rounded** `max_scaled`
from the **unrounded** product `accS*sm_log2`, leaving the rounding error of
`m_cur*sm_log2` (~ -1.7e-7) instead of 0. So the softmax max comes out **0.99999988,
not 1.0**. Then the P-requant block scale `mx_scale_exp(0.99999988)` =
`floor(log2)-8` = `-9` (one too low vs `-8` for exactly 1.0), the max element scales
to `1.0·2^9 = 512`, **saturates to e4m3 max 448**, and dequantizes to
`448·2^-9 = 0.875`. Every row whose softmax-argmax 32-key block has amax == 1.0 comes
out uniformly **×7/8** (~37/128 rows on random data).

**Fix:** subtract the max BEFORE scaling — `p = exp2f((accS - m_sub)*sm_log2)`,
`m_sub = (m_cur==-inf)?0:m_cur`. `(a-b)*c` is **not** an FMA pattern (the sub feeds
the mul, not mul→add), so at the argmax `(m_cur-m_cur)*sm = 0` exactly →
`exp2f(0)=1.0` → se=-8 → no saturation. **`a*c - b*c` is NOT FMA-safe even when a==b.**

### Why the bit-exact self-replay was BLIND to it (the meta-lesson)

The self-replay reference re-quantizes the **device's own dumped P** with the
**identical** `mx_scale_exp` rule: device computes 0.99999988 → se=-9 → 0.875, host
re-quant of that same 0.99999988 → se=-9 → 0.875 → **they agree → "PASS"**. A test
that shares the quantization rule with the kernel cannot catch a bug in that rule's
*input*. Only the **independent torchao oracle** (`tests/csrc/mxfp8_attn_ext.cu` +
`test_mxfp8_prefill.py::test_attn_matches_torchao`, fp64 softmax where the max is
exactly 1.0) exposed it — at **25% of elements wrong**. Rule: a self-consistency test
and an independent-oracle test catch **disjoint** bug classes; S3 needed both.
(Non-argmax blocks with amax mantissa in `[1.75,2)` also saturate, but torchao and
the kernel both use the floor rule and saturate identically, so they agree —
`cutlass::float_e4m3_t` matches `to_mx` element-for-element. Only the argmax block was
special because FMA nudged amax just *below* the 1.0 power-of-2 boundary.)

---

## V's scale-factor (SFV) layout: K's SF layout only *coincides* with V's at a single 128×128 block

**TL;DR:** In PV, V is the **B operand** shaped `[head_dim, keys]`, block-scaled
**along keys**. Its scale-factor (ue8m0) tensor must be laid out as
`tile_atom_to_shape_SFB(_, head_dim, keys)` = `tile_to_shape(SfAtom, (head_dim, keys))`.
Do **not** reuse K's SF layout (`tile_atom_to_shape_SFA(keys, _, head_dim)` =
`(keys, head_dim)`, scale along head_dim) for V. For one 128×128 block (head_dim ==
kBlockN) the two are byte-identical, so a single-tile test passes — but they are
**transposed**, and the moment V spans more than one n_block the per-block SFV
slice for keys ≥ 128 reads the wrong gmem region → ue8m0 byte `0` = `2^-127` ≈ 0 →
`V · scale ≈ 0` → that block's PV ≈ 0.

### The symptom (and how it hides)

Multi-n_block: **only the first n_block's PV produces output; blocks 1+ give
accB ≈ 0.** Everything upstream is fine — QK, softmax, `row_sum`, LSE, P-requant,
the PV *data* operands `tOrP`/`tOrV`, and **P's** scale `tOrSFP` are all correct
for every block. Only **`tOrSFV` is nonzero for block 0 and ≈ 1e-35 for blocks
1,2,3** (per-block dump). The whole thing is an **input-side gmem-layout bug**, not
a quantization-math or consumer-smem bug — the consumer's per-block `SmemLayoutSFV`
is fine precisely because head_dim == kBlockN makes the per-block shape coincide.

The reason this single bug ate a long debugging session is a **test-data trap**:
adversarial data that makes the running softmax max grow fast (to "exercise" the
online rescale) drives `scores_scale = exp2((m_prev−m_cur)·scale)` to **fp32
underflow**, collapsing the output onto a single n_block — and if that surviving
block is block 0 (the only one whose SFV is correct), the end-to-end check
**passes** and the bug stays invisible. Switching to FLAT data (every block
contributes materially) exposed it immediately.

### The rule

- Build each operand's SF from the function that matches its **role and
  scale-axis**: SFA (`tile_to_shape(SfAtom,(M,K))`) for an A operand scaled along
  its K; SFB (`(N,K)`) for a B operand scaled along its K. V-in-PV is B,
  scaled along keys → SFB over `(head_dim, keys)`. Never borrow another operand's
  SF layout just because the per-tile shapes look equal.
- A single-tile (one n_block, head_dim == kBlockN) test **cannot** catch a
  transposed-SF bug — the layouts coincide there. Always validate with **≥2
  n_blocks** and data where **every block contributes** to O (not max-collapsed
  onto one block). Dump per-block `sum|operand|` / `sum|SF|` to localize which
  operand of which block is zero.
- This is the same generalization needed for head_dim ≠ kBlockN (where V's SF
  shape stops coinciding with K's even for a single block) — do it once, correctly,
  for V.

---

## P→PV requant: QK-C and PV-A are NOT thread-local aligned (and nvfp4 only *looks* like it is)

**TL;DR:** In the fused kernel, the softmax probabilities P come out of QK as the
**C accumulator** and must go back in as the **PV A operand**. For the mxf8
`SM120_16x8x32_TN_VS` atom these two layouts do **not** match per thread, so P
cannot be reused in-register by a simple relabel — it has to be physically
redistributed. The current kernel does this through an smem transpose buffer
(`tests/s2_pv.cu`). Don't assume P "just stays in registers."

### The mismatch

For the m16n8k32 atom, per thread:
- **QK-C** N(key) micro-layout = "2 adjacent keys, stride-8 comb" (FP32 accumulate,
  n8 → 2 cols/thread).
- **PV-A** K(key) micro-layout = "4 adjacent keys" (FP8 A operand, k32 → 4 bytes/thread).

The 2-vs-4 granularity is intrinsic to the hardware instruction's TV layouts.
Probe (`/tmp/locality.cu`, `partition_C` vs `partition_A` over the same
(q,key) identity tensor): **256/256 threads have different {(q,key)} sets.**

### The nvfp4 trap — it looks aligned, it isn't

Reading SageAttention you'll see `SM120_16x32x64_TN_VS_NVFP4` and may conclude
"their n32 atom makes QK-C and PV-A line up, so P is register-resident for free,
and mxf8 is uniquely unlucky." **Not true on either count:**

- `SM120_16x32x64_TN_VS_NVFP4` is **not a hardware MMA**. It's a hand-written
  *macro* atom (`cute_extension.h`) that issues **4× the real
  `mma.sync…m16n8k64` instruction** stacked in N, sharing one A operand and one
  `scale_vec::4X` SFA. Every hardware blockscaled MMA on sm120 is m16n8
  (`cute/arch/mma_sm120.hpp`: only `SM120_16x8x32` / `SM120_16x8x64`). Stacking
  instructions tiles the per-instruction TV layout — it cannot change it.
- The same probe on the nvfp4 macro: **also 256/256 mismatching.** nvfp4 is
  *not* thread-local aligned either. SageAttention keeps P in registers via the
  **V-permutation trick** (fill `tOrP` thread-locally, which permutes the key
  axis by some π, then load V in that same π order so `Σ P[π(k)]·V[π(k)]` is
  still correct) plus its quad `__shfl_xor` on the SF — not free alignment.

### The upside (and the rule)

The probe's useful half: for **both** mxf8 and nvfp4 the QK-C→PV-A redistribution
is **entirely within a 4-lane quad** (same q-rows; the 128 keys of those rows just
re-partition among the 4 quad lanes). So **smem is not required** — an intra-quad
`__shfl` (move only P) or the V-permutation trick (P stays thread-local, V loaded
key-permuted) both work and skip the smem write + named-barrier + ldmatrix.

**Rule:** the smem transpose buffer is the simplest *correct* path — keep it as the
oracle. The intra-quad shuffle / V-permutation is a real perf win (drops a
warpgroup-serializing barrier) but is an optimization to layer on *after* the
pipeline is numerically correct, validated bit-exact against the smem version.

---

## Persistent tile scheduling is neutral-to-harmful for a non-warp-specialized mainloop

**TL;DR:** Don't add a persistent tile scheduler to the option-B (non-WS, cp.async)
mainloop. Launch plain non-persistent (`grid = num_q_blocks × heads × batch`) and
let the hardware block scheduler balance the load. It already balances causal
(triangular) work as well as a software dynamic scheduler — for free.

### Why this is a trap

SageAttention3 / FlashAttention-3 use a **persistent** kernel (grid = #SMs, each
CTA loops over tiles via a `tile_scheduler`). It's tempting to copy that. But their
persistent design exists for reasons that **don't apply to a non-WS kernel**:

1. Their warp-specialized CTAs already occupy **1 block/SM**, so `grid = #SMs` *is*
   full occupancy.
2. The persistent loop **amortizes an expensive prologue** (TMA descriptor setup,
   mbarrier init, producer/consumer warp setup) across many tiles.

A non-WS cp.async kernel has a cheap prologue (nothing to amortize) and occupancy
> 1 block/SM — so persistent scheduling buys nothing and can actively hurt.

### Measured A/B (RTX 5060 Ti, 36 SMs) — `bench/sched_ab.cu`

Same tunable per-tile compute kernel under three schedulers from SageAttention's
`tile_scheduler.h`, swept over 1–32 waves and {uniform, causal-triangular} load:

| scheduler | uniform | causal (typical) | causal (worst) |
|---|---|---|---|
| non-persistent (`grid=num_tiles`, HW scheduler) | 1.00x (baseline) | 1.00x | 1.00x |
| **static**-persistent (`grid=num_sm×occ`, round-robin) | ~1.1x | ~1.1x | **1.46x slower** |
| dynamic-persistent (`grid=num_sm×occ`, atomicAdd) | ~1.0x | ~1.0x | ~1.0x |

- **non-persistent ≈ dynamic-persistent ≈ optimal**, robustly, across all configs.
  The HW block scheduler load-balances triangular causal work for free.
- **static-persistent is strictly dominated** — never faster, up to 1.46x slower.
  SageAttention wires in exactly the *static* one (`launch.h:42`).
- The 1.46x worst case is **period aliasing**: when the causal work-period `G`
  (q-blocks per head) divides the persistent grid stride (`G=8 | grid 216`), every
  tile a given CTA processes lands on the same qblock, so one CTA inherits the
  entire heavy tail and gates the launch. Non-divisor `G` (7/5/13) → only ~1.1x.
  non-/dynamic-persistent are immune (verified by sweeping `G`).

### ncu evidence — why dynamic-persistent doesn't beat non-persistent, and why static is slower

Profiled the three kernels at the heavy causal config (1152 tiles, 32 waves,
`./bench/sched_ab 8 profile`), RTX 5060 Ti:

| kernel | duration | SM throughput (% peak, elapsed) | achieved occupancy (elapsed) | instructions |
|---|---|---|---|---|
| non-persistent | 2.328 ms | **86.8%** | **83.0%** | 747,424,512 |
| static-persistent | 3.252 ms (1.40x) | **62.1%** | **47.6%** | 747,465,408 |
| dynamic-persistent | 2.297 ms | **87.9%** | **84.6%** | 747,524,376 |

- **All three execute the same instruction count** (within 0.02%). The runtime
  difference is *entirely* whether the SMs stay fed, not extra work. The dynamic
  scheduler's atomic-counter overhead is negligible (+0.01% instructions).
- **non-persistent ≈ dynamic-persistent**: both hold ~83–85% achieved occupancy and
  ~87% SM throughput — the SMs are already saturated and balanced. The HW block
  scheduler *is* a dynamic greedy load-balancer; a software atomic scheduler just
  re-does what the hardware already does → no slack to recover → ~1.0x. (dynamic is
  a marginal 1.3% faster here only because 216 long-lived CTAs pay less launch/retire
  turnover than 1152 short ones at high wave counts; at low waves it's *slower*.)
- **static-persistent is slower purely from idle tail**: same work, but achieved
  occupancy collapses 83%→48% and throughput 87%→62%. The duration ratio
  (3.252/2.328 = 1.40x) equals the throughput ratio (86.8/62.1 = 1.40x) almost
  exactly — the lost time is SMs sitting idle after their light tiles finish while
  heavy-tile CTAs grind on; static round-robin can't hand idle SMs the heavy work.

(The FMA pipe sits at ~43% because the microbench's burn loop is a single-thread
dependency chain, latency-bound; irrelevant to the scheduling conclusion — read the
SM-throughput and occupancy rows, which show the SMs are kept busy issuing.)

### Real-kernel confirmation (WS+TMA) — persistent IS worth ~15% on balanced load

The FMA microbench above **explicitly could not** measure persistent's real
upside: amortizing the expensive WS+TMA prologue (TMA descriptor prefetch,
mbarrier init, producer/consumer reg dealloc/alloc) across many tiles. So I
built the **actual SageAttention3 nvfp4 WS+TMA kernel** and A/B'd it for real —
toggling `launch.h:41/42` (`SingleTileScheduler` vs `StaticPersistentTileScheduler`)
and rebuilding `fp4attn_cuda` between runs. `bench/bench_sage3_real.py`, RTX
5060 Ti, B=4 H=32 D=128 fp16. Output is **bit-identical** between the two
schedulers (1-cos vs SDPA = 0.0181 non-causal / 0.254 causal, same to 4 digits)
— confirms the scheduler is numerically transparent, so the timing delta is
purely scheduling.

Full-op TFLOPS (higher = better):

| workload | seqlen | persistent | non-persistent | winner |
|---|---|---|---|---|
| non-causal (balanced)   | 8192  | 152.1 | 143.1 | **persistent +6.3%** |
| non-causal (balanced)   | 16384 | 170.3 | 160.6 | **persistent +6.0%** |
| causal (triangular)     | 8192  | 110.9 | 114.1 | non-persistent +2.9% |
| causal (triangular)     | 16384 | 135.2 | 140.8 | non-persistent +4.0% |

ncu, **attention kernel only** (non-causal, L=4096) — the smoking gun for
prologue amortization:

| metric | persistent | non-persistent |
|---|---|---|
| grid | **36 CTAs** (= #SM) | **4096 CTAs** (m×h×b) |
| waves / SM | 1 | 113.8 |
| achieved occupancy | 20.7% | 20.0% |
| SM throughput | **57.2%** | 48.7% |
| kernel duration | **5.09 ms** | 5.97 ms (**+17%**) |

- **Occupancy is identical (~20% = 1 block/SM)** — both run one heavy WS CTA per
  SM. So unlike the microbench's *static* collapse (occupancy 83→48%), here the
  difference is **not** occupancy. Same work, same residency.
- The win is **SM throughput: 57.2% vs 48.7%.** The ratio 57.2/48.7 = 1.17 equals
  the duration ratio 5.97/5.09 = 1.17 exactly. The persistent kernel keeps the
  SMs ~17% busier because it pays the prologue **36 times** (one per resident CTA,
  then loops ~114 tiles each) instead of **4096 times** (a fresh CTA launch +
  prologue per tile, streamed 113.8 waves deep). That is prologue + launch/retire
  amortization, made visible.
- **Causal flips the sign by a few %:** static round-robin hands one CTA the heavy
  triangular tail (the microbench's idle-tail effect), costing ≤4% vs the HW
  scheduler — but nowhere near the synthetic 1.46x, because real per-tile work is
  far heavier than a burn loop. A **dynamic** persistent scheduler would keep the
  prologue win *and* balance the tail; SageAttention ships *static*, leaving the
  causal few-% on the table.

**This does not contradict the non-WS finding above — it completes it.** The
microbench verdict ("persistent neutral-to-harmful") is correct *and scoped to a
cheap-prologue non-WS kernel*. For a heavy-prologue WS+TMA kernel the prologue
amortization is real and dominates on balanced load (~15% at the kernel level,
~6% end-to-end once the identical quant/group-mean pre-kernels dilute it).

### Why static-persistent *loses* on causal — it's two orthogonal effects

Counter-intuitive: causal (triangular) load is exactly where you'd expect a
scheduler to help, yet static-persistent loses ~2.5–4% there. Profiled the
controlled 2×2 (`bench/profile_sched_2x2.sh`, swaps prebuilt .so variants, L=4096)
plus a causal L-sweep. **The clean, clock-independent probe is `sm__cycles_active`
— the fraction of elapsed time an SM has *any* warp resident, i.e. "not idle".**
(Ignore ncu's *absolute* kernel ms here: it locks clocks to base, and for a ~3%
effect its single-shot durations are unreliable — at L=8192 they even flipped the
sign vs the averaged benchmark. Trust the averaged benchmark for timing and ncu
only for the `pct_of_peak_sustained_elapsed` *fractions*, which are ratios within
one locked-clock run and so clock-independent.)

| variant | mask | sm__cycles_active (not-idle) | sm__throughput |
|---|---|---|---|
| persistent     | non-causal | **99.5%** | 57.3% |
| non-persistent | non-causal | 98.2% | 48.2% |
| persistent     | causal | **89.0%** | 45.9% |
| non-persistent | causal | **94.4%** | 47.9% |

Causal, sweeping L (cycles_active / throughput, %):

| L | persistent | non-persistent |
|---|---|---|
| 4096  | 89.0 / 45.9 | 94.4 / 47.9 |
| 8192  | 92.4 / 49.0 | 94.2 / 51.0 |
| 16384 | 97.5 / 52.3 | 99.1 / 55.9 |

Read it as **two orthogonal effects** that the non-causal/causal split cleanly
separates:

1. **Prologue / pipeline-fill amortization** — always favors persistent. Visible
   non-causal as throughput 57.3 vs 48.2 (+9 pts) at ~equal, ~99% occupancy-of-time.
   36 prologues vs 4096.
2. **Load balancing** — favors whoever can rebalance the heavy triangular tail.
   Static round-robin **can't**: it fixes each CTA's tile set up front, so when a
   CTA exhausts its (lighter) tiles it goes idle while the CTA holding the heavy
   tail grinds on. That's the **`sm__cycles_active` drop to 89–97% on causal**,
   always below non-persistent's 94–99%. The hardware block scheduler (what
   non-persistent rides) is a greedy dynamic balancer — it hands an idle SM the
   next pending tile, heavy or light, and stays 94–99% fed.

Non-causal: only effect (1) is in play → persistent wins (~15% kernel). Causal:
effect (2)'s idle tail costs static-persistent more than effect (1)'s prologue
edge saves → persistent nets ~2.5–4% slower (and the throughput gap widens with
L: 1.9 → 2.0 → 3.6 pts). The user's intuition was right — causal *is* a
load-balancing problem — but **static-persistent is the wrong tool for it; you
need a dynamic scheduler to claim that win.**

Caveat that matters for us: SageAttention's `DynamicPersistentTileScheduler` is a
**non-functional stub** — its `get_next_work` is identical to the static one
(`tile_idx + gridDim.x`) and the `tile_count_semaphore` it stores is never read.
So the "best of both" (prologue amortization + greedy balancing) isn't actually
available in this codebase without implementing the atomic tile fetch ourselves.

### Rules

- **Option B (non-WS) mainloop: start non-persistent.** Zero-gain to do otherwise,
  plus the static-persistent aliasing tail risk.
- If we later go **option A (WS + TMA)**, persistent is justified — **measured**
  ~15% kernel-level / ~6% end-to-end on balanced load from prologue amortization
  at occupancy=1 (see real-kernel section above). But a *static* persistent
  scheduler gives the prologue win back on causal via the idle tail (−2.5–4%).
- **FA3 confirms our 2×2 exactly** (`flash-attention/hopper/flash_fwd_launch_
  template.h:61-76`): it picks **StaticPersistent only for dense non-causal**, the
  **real DynamicPersistent for dense causal/local**, **VarlenDynamic for varlen**,
  and SingleTile for split/low-work. So StaticPersistent is not vestigial — its
  niche is precisely the no-imbalance case we measured it winning. Mirror this map.
- The **real dynamic scheduler is more than an atomic counter**: FA3's does **LPT**
  (issue heaviest tiles first, `block = num_m_blocks-1-block`) + **L2 swizzling**
  (head×batch sections sized to keep K/V in L2). SageAttention's `Dynamic…` is a
  stub (no atomic, no LPT) — don't copy it; copy FA3's. For FlashInfer's **ragged
  BatchPrefill the relevant one is the varlen dynamic scheduler** (batch prefix-sums
  + LPT), the hardest of the four — plan it early. If causal dominates and we want
  to defer that work, plain non-persistent is within a few % and far simpler.

### Two methodology gotchas (caught while running the A/B)

1. **A persistent kernel must launch `grid = num_sm × cudaOccupancyMaxActiveBlocks
   PerMultiprocessor`, not `num_sm`.** A light kernel has occupancy > 1 (here 6
   blocks/SM); using `grid = num_sm` starves it to 1/6 load, and you end up
   measuring occupancy loss (a fake ~2x slowdown), not scheduling. FA3/SageAttention
   get away with `grid = num_sm` only because their heavy WS CTAs are already 1/SM.

2. **This microbench cannot measure persistent's real benefit** for FA3/SageAttention
   (amortizing the WS+TMA prologue). It therefore argues against persistent for a
   *cheap-prologue non-WS* kernel **only** — not against option A's persistent.

### Side fix

SageAttention's `sageattn3/blackwell/launch.h:89` hardcoded `num_sm = 170` (the
RTX 5090's SM count). Patched to query
`getCurrentDeviceProperties()->multiProcessorCount` so the persistent grid is
correct on any GPU (36 on the 5060 Ti) — otherwise persistent over-subscribes
170 CTAs onto 36 SMs and the A/B is polluted before it starts.

## sm120 smem optin limit is **101376B** (99KB), NOT 228KB — RTX 5060 Ti (GB206) has 100KB smem/SM total. `cudaFuncSetAttribute(MaxDynamicSharedMemorySize, >101376)` fails with `invalid argument` (NOT `invalid value` — misleading). Any fused/multi-ring design over ~96KB of data + SF + barriers simply does not launch. (Killed the 2-stage fused dvdk2@kv128: 105472B; had to drop the Qt/Dt ring to single-stage to fit.)
