#!/usr/bin/env bash
# SKAB (Skoltech Anomaly Benchmark) + TEP Braatz files.
set -e
ROOT=${INDUSTRIAL_TS_DATA:-data/raw}; mkdir -p "$ROOT/TEP"
[ -d "$ROOT/SKAB" ] || git clone --depth 1 https://github.com/waico/SKAB "$ROOT/SKAB"
[ -d "$ROOT/TEP/braatz" ] || git clone --depth 1 https://github.com/camaramm/tennessee-eastman-profBraatz "$ROOT/TEP/braatz"
echo "export INDUSTRIAL_TS_DATA=$ROOT"
