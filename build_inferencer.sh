#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
nvcc -O3 -std=c++17 -arch=sm_87 inferencer.cu -lpcre2-8 -Xcompiler -pthread -o inferencer
