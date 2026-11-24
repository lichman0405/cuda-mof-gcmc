#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# Locate nvcc across PATH and common CUDA / NVHPC installation locations.
find_nvcc() {
    if command -v nvcc >/dev/null 2>&1; then
        command -v nvcc
        return 0
    fi
    local cand
    for cand in /usr/local/cuda/bin/nvcc /usr/local/cuda-*/bin/nvcc "$HOME"/nvhpc/Linux_x86_64/*/cuda/*/bin/nvcc; do
        if [[ -x "$cand" ]]; then
            echo "$cand"
            return 0
        fi
    done
    return 1
}

NVCC_BIN="$(find_nvcc || true)"
if [[ -z "$NVCC_BIN" ]]; then
    echo "Error: Could not locate nvcc. Add CUDA bin directory to PATH." >&2
    exit 1
fi

CUDA_BIN_DIR="$(dirname "$NVCC_BIN")"
CUDA_HOME_DIR="$(dirname "$CUDA_BIN_DIR")"
case ":$PATH:" in
    *":$CUDA_BIN_DIR:"*) ;;
    *) export PATH="$CUDA_BIN_DIR:$PATH" ;;
esac
if [[ -d "$CUDA_HOME_DIR/lib64" ]]; then
    case ":${LD_LIBRARY_PATH:-}:" in
        *":$CUDA_HOME_DIR/lib64:"*) ;;
        *) export LD_LIBRARY_PATH="$CUDA_HOME_DIR/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
    esac
fi

# Auto-detect CUDA compute capability (respect CUDA_ARCH override, e.g. CUDA_ARCH=sm_86).
if [[ -z "${CUDA_ARCH:-}" ]]; then
    CC_NODOT=""
    if command -v nvidia-smi >/dev/null 2>&1; then
        CC_NODOT="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n 1 | tr -d ' .\r')"
    fi
    if [[ -n "$CC_NODOT" ]]; then
        CUDA_ARCH="sm_${CC_NODOT}"
    else
        CUDA_ARCH="sm_86"
    fi
fi

mkdir -p build

TARGET="${1:-core}"

build_cli() {
    echo "Building build/mof_gcmc_cli for ${CUDA_ARCH} (${NVCC_BIN})..."
    "$NVCC_BIN" -O3 -std=c++17 -arch="${CUDA_ARCH}" -Iinclude \
        -Xcompiler "-O3" \
        src/cif_loader.cpp src/unified_grid_builder.cu src/unified_gcmc_kernel.cu src/main_cli.cu \
        -o build/mof_gcmc_cli
    echo "Build succeeded: build/mof_gcmc_cli"
}

build_proposer() {
    echo "Building build/libmof_gcmc_proposer.so for ${CUDA_ARCH} (${NVCC_BIN})..."
    "$NVCC_BIN" -O3 -std=c++17 -arch="${CUDA_ARCH}" -Iinclude --shared \
        -Xcompiler "-O3 -fPIC" \
        src/cif_loader.cpp src/unified_grid_builder.cu src/unified_gcmc_kernel.cu src/proposer_dll.cu \
        -o build/libmof_gcmc_proposer.so
    echo "Build succeeded: build/libmof_gcmc_proposer.so"
}

build_single() {
    echo "Building build/mof_gcmc for ${CUDA_ARCH}..."
    "$NVCC_BIN" -O3 -std=c++17 -arch="${CUDA_ARCH}" -Iinclude \
        -Xcompiler "-O3" \
        src/cif_loader.cpp src/legacy/grid_builder.cu src/legacy/gcmc_kernel.cu src/legacy/main.cu \
        -o build/mof_gcmc
    echo "Build succeeded: build/mof_gcmc"
}

build_mix() {
    echo "Building build/mof_gcmc_mix for ${CUDA_ARCH}..."
    "$NVCC_BIN" -O3 -std=c++17 -arch="${CUDA_ARCH}" -Iinclude \
        -Xcompiler "-O3" \
        src/cif_loader.cpp src/legacy/mix_grid_builder.cu src/legacy/mix_gcmc_kernel.cu src/legacy/main_mix.cu \
        -o build/mof_gcmc_mix
    echo "Build succeeded: build/mof_gcmc_mix"
}

build_v2() {
    echo "Building build/mof_gcmc_v2 for ${CUDA_ARCH}..."
    "$NVCC_BIN" -O3 -std=c++17 -arch="${CUDA_ARCH}" -Iinclude \
        -Xcompiler "-O3" \
        src/cif_loader.cpp src/legacy/grid_builder.cu src/legacy/grid_builder_v2.cu src/legacy/gcmc_kernel_v2.cu src/legacy/main_v2.cu \
        -o build/mof_gcmc_v2
    echo "Build succeeded: build/mof_gcmc_v2"
}

case "$TARGET" in
    core)
        build_cli
        build_proposer
        ;;
    cli)
        build_cli
        ;;
    proposer)
        build_proposer
        ;;
    single)
        build_single
        ;;
    mix)
        build_mix
        ;;
    v2)
        build_v2
        ;;
    legacy)
        build_single
        build_mix
        build_v2
        ;;
    all)
        build_cli
        build_proposer
        build_single
        build_mix
        build_v2
        ;;
    *)
        echo "Unknown target '$TARGET'. Usage: $0 [core|cli|proposer|legacy|single|mix|v2|all]" >&2
        exit 1
        ;;
esac
