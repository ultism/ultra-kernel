#!/usr/bin/env bash
# Controlled 2x2 ncu comparison of the real SageAttention3 WS+TMA attention
# kernel: {persistent, non-persistent} x {causal, non-causal}, at a fixed shape
# (B=4 H=32 D=128 L=4096). Swaps prebuilt .so variants into place so no rebuild
# is needed. Purpose: explain WHY static-persistent LOSES on causal even though
# causal (triangular) is supposedly where a scheduler should help.
#
# The key probe is sm__cycles_active (fraction of elapsed time the SMs are doing
# *anything*): a load-imbalance idle tail shows up as this dropping. Paired with
# sm__throughput (issue/pipe utilization while active).
set -euo pipefail
SAGE=/root/fa-blackwell/tmp/SageAttention/sageattention3_blackwell
CANON="$SAGE/fp4attn_cuda.cpython-312-x86_64-linux-gnu.so"
PY=/root/miniconda3/envs/torchao-dev/bin/python
BENCH=/root/fa-blackwell/bench
METRICS=launch__grid_size,launch__waves_per_multiprocessor,sm__warps_active.avg.pct_of_peak_sustained_elapsed,sm__cycles_active.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.sum

for variant in persistent nonpersistent; do
  cp "$BENCH/fp4attn_${variant}.so" "$CANON"
  for mask in noncausal causal; do
    echo "############## ${variant} / ${mask} ##############"
    PYTHONPATH=$SAGE ncu --target-processes all \
      --kernel-name "regex:compute_attn_ws" -s 3 -c 1 \
      --metrics "$METRICS" \
      "$PY" /tmp/sage_oneshot.py "$mask" 2>&1 \
      | grep -E "grid_size|waves_per|warps_active|sm__cycles_active|sm__throughput|tensor_cycles|time_duration"
  done
done
# leave persistent (as-shipped) in place
cp "$BENCH/fp4attn_persistent.so" "$CANON"
echo "DONE (restored persistent .so)"
