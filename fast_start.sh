#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

CHECK_ONLY=0
SKIP_BUILD=0
REBUILD=0
SKIP_TEST=0
FULL_TEST=0
SMOKE_MLIP=0
ARCH_OVERRIDE=""
CUDA_INDEX=""
TORCH_INDEX_URL="${TORCH_INDEX_URL:-}"
VENV_DIR=".venv-mlip"
PY_VER="3.12"
GPU_ID=""

usage() {
    cat <<'EOF'
Usage: ./fast_start.sh [options]
  --check-only        Inspect GPU, CUDA, uv/Python, packages, and binaries without installing/building
  --skip-build        Do not compile CUDA binaries
  --rebuild           Force clean rebuild of build/mof_gcmc_cli and build/libmof_gcmc_proposer.so
  --skip-test         Skip post-setup verification tests
  --full-test         Run full non-MLIP pytest suite (including slow MC distribution parity tests)
  --smoke-mlip        Also run a short live auto_mlip smoke simulation after pytest
  --arch sm_XX        Override CUDA compute capability (default: auto-detect via nvidia-smi)
  --cuda-index cuXXX  Override PyTorch wheel tag (e.g. cu124, cu128; default: auto from driver)
  --torch-index-url U Override full PyTorch wheel index URL (e.g. https://mirror.nju.edu.cn/pytorch/whl/cu124)
  --venv DIR          Virtual environment directory (default: .venv-mlip)
  --python VER        Python version for uv venv (default: 3.12)
  --gpu ID            Pin CUDA_VISIBLE_DEVICES to GPU index ID (default: GPU with lowest memory.used)
  -h, --help          Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --check-only) CHECK_ONLY=1; shift ;;
        --skip-build) SKIP_BUILD=1; shift ;;
        --rebuild) REBUILD=1; shift ;;
        --skip-test) SKIP_TEST=1; shift ;;
        --full-test) FULL_TEST=1; shift ;;
        --smoke-mlip) SMOKE_MLIP=1; shift ;;
        --arch) ARCH_OVERRIDE="${2:?--arch requires a value}"; shift 2 ;;
        --cuda-index) CUDA_INDEX="${2:?--cuda-index requires a value}"; shift 2 ;;
        --torch-index-url) TORCH_INDEX_URL="${2:?--torch-index-url requires a value}"; shift 2 ;;
        --venv) VENV_DIR="${2:?--venv requires a value}"; shift 2 ;;
        --python) PY_VER="${2:?--python requires a value}"; shift 2 ;;
        --gpu) GPU_ID="${2:?--gpu requires a value}"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
    esac
done

for d in "$HOME/.local/bin" "$HOME/.cargo/bin" "/usr/local/cuda/bin"; do
    if [[ -d "$d" ]]; then
        case ":$PATH:" in
            *":$d:"*) ;;
            *) export PATH="$d:$PATH" ;;
        esac
    fi
done

step() { printf '\n==> %s\n' "$1"; }
ok()   { printf '  [OK] %s\n' "$1"; }
warn() { printf '  [WARN] %s\n' "$1"; }
miss() { printf '  [MISSING] %s\n' "$1"; }

MISSING_ITEMS=()

# 1. GPU & NVIDIA Driver
step "1/6 Checking NVIDIA GPU & Driver"
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "Error: nvidia-smi not found. Install NVIDIA GPU driver first." >&2
    exit 1
fi

GPU_ROWS="$(nvidia-smi --query-gpu=index,name,memory.total,memory.used,compute_cap --format=csv,noheader,nounits)"
GPU_COUNT="$(printf '%s\n' "$GPU_ROWS" | wc -l | tr -d ' ')"

if [[ -n "$GPU_ID" ]]; then
    export CUDA_VISIBLE_DEVICES="$GPU_ID"
elif [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    BEST_GPU="$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t',' -k2 -n | head -n 1 | cut -d',' -f1 | tr -d ' ')"
    if [[ -n "$BEST_GPU" ]]; then
        export CUDA_VISIBLE_DEVICES="$BEST_GPU"
    fi
fi

SEL_GPU="${CUDA_VISIBLE_DEVICES:-0}"
FIRST_SEL_ID="${SEL_GPU%%,*}"
SEL_LINE="$(printf '%s\n' "$GPU_ROWS" | awk -F',' -v id="$FIRST_SEL_ID" '{gsub(/^ +| +$/,"",$1); if($1==id){print; exit}}')"
if [[ -z "$SEL_LINE" ]]; then
    SEL_LINE="$(printf '%s\n' "$GPU_ROWS" | head -n 1)"
fi

GPU_NAME="$(printf '%s\n' "$SEL_LINE" | awk -F',' '{gsub(/^ +| +$/,"",$2); print $2}')"
GPU_MEM_MIB="$(printf '%s\n' "$SEL_LINE" | awk -F',' '{gsub(/^ +| +$/,"",$3); print $3}')"
CC_RAW="$(printf '%s\n' "$SEL_LINE" | awk -F',' '{gsub(/^ +| +$/,"",$5); print $5}')"
CC_NODOT="${CC_RAW//./}"

if [[ -n "$ARCH_OVERRIDE" ]]; then
    export CUDA_ARCH="$ARCH_OVERRIDE"
elif [[ -z "${CUDA_ARCH:-}" ]]; then
    export CUDA_ARCH="sm_${CC_NODOT:-86}"
fi

DRIVER_CUDA="$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -n 1)"
DRIVER_CUDA="${DRIVER_CUDA:-12.4}"

if [[ -z "$CUDA_INDEX" ]]; then
    if awk "BEGIN{exit !(${CC_RAW:-0} >= 12.0 || ${DRIVER_CUDA:-0} >= 12.8)}"; then
        CUDA_INDEX="cu128"
    elif awk "BEGIN{exit !(${DRIVER_CUDA:-0} >= 12.4)}"; then
        CUDA_INDEX="cu124"
    elif awk "BEGIN{exit !(${DRIVER_CUDA:-0} >= 12.1)}"; then
        CUDA_INDEX="cu121"
    else
        CUDA_INDEX="cu118"
    fi
fi

if [[ -z "$TORCH_INDEX_URL" ]]; then
    TORCH_INDEX_URL="https://download.pytorch.org/whl/${CUDA_INDEX}"
fi

ok "Detected ${GPU_COUNT} GPU(s); using CUDA_VISIBLE_DEVICES=${SEL_GPU} (${GPU_NAME}, ${GPU_MEM_MIB} MiB, compute ${CC_RAW}, target ${CUDA_ARCH})"
ok "Driver max CUDA ${DRIVER_CUDA} -> PyTorch index ${CUDA_INDEX} (${TORCH_INDEX_URL})"

# 2. CUDA Compiler & C++ Host Toolchain
step "2/6 Checking CUDA Toolkit (nvcc) & g++"
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
if [[ -n "$NVCC_BIN" ]]; then
    NVCC_DIR="$(dirname "$NVCC_BIN")"
    case ":$PATH:" in
        *":$NVCC_DIR:"*) ;;
        *) export PATH="$NVCC_DIR:$PATH" ;;
    esac
    NVCC_VER="$("$NVCC_BIN" --version | tail -n 1)"
    ok "nvcc: ${NVCC_BIN} (${NVCC_VER})"
else
    miss "CUDA Toolkit (nvcc) not found."
    MISSING_ITEMS+=("CUDA Toolkit (nvcc)")
fi

if command -v g++ >/dev/null 2>&1; then
    ok "g++: $(g++ --version | head -n 1)"
else
    miss "g++ not found (install build-essential)."
    MISSING_ITEMS+=("g++")
fi

# 3. uv / Python & Virtual Environment
step "3/6 Checking Python, uv & Virtual Environment (${VENV_DIR})"
UV_BIN=""
if command -v uv >/dev/null 2>&1; then
    UV_BIN="$(command -v uv)"
    ok "uv: ${UV_BIN} ($("$UV_BIN" --version))"
    export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
    if [[ -z "${UV_DEFAULT_INDEX:-}" ]]; then
        for cfg in "$HOME/.config/pip/pip.conf" "$HOME/.pip/pip.conf"; do
            if [[ -f "$cfg" ]]; then
                PIP_IDX="$(sed -n 's/^[[:space:]]*index-url[[:space:]]*=[[:space:]]*\([^[:space:]#]*\).*/\1/p' "$cfg" | head -n 1)"
                if [[ -n "$PIP_IDX" ]]; then
                    export UV_DEFAULT_INDEX="$PIP_IDX"
                    ok "Inherited uv PyPI mirror from ${cfg}: ${UV_DEFAULT_INDEX}"
                    break
                fi
            fi
        done
    fi
else
    warn "uv not found; falling back to system python3 -m venv / pip."
fi

SYS_PY=""
if command -v python3 >/dev/null 2>&1; then
    SYS_PY="$(command -v python3)"
    ok "system python3: $("$SYS_PY" --version)"
fi

if [[ "$VENV_DIR" = /* ]]; then
    VENV_PATH="$VENV_DIR"
else
    VENV_PATH="$ROOT/$VENV_DIR"
fi
VENV_PY="$VENV_PATH/bin/python"

if [[ -x "$VENV_PY" ]]; then
    ok "Virtual env present: ${VENV_PATH} ($("$VENV_PY" --version))"
else
    miss "Virtual env missing at ${VENV_PATH}."
    if [[ "$CHECK_ONLY" -eq 1 ]]; then
        MISSING_ITEMS+=("Virtual env ${VENV_DIR}")
    else
        if [[ -n "$UV_BIN" ]]; then
            echo "  Creating ${VENV_PATH} with uv (Python ${PY_VER})..."
            if ! "$UV_BIN" venv --python "$PY_VER" "$VENV_PATH"; then
                warn "uv venv --python ${PY_VER} failed; retrying with default Python..."
                "$UV_BIN" venv "$VENV_PATH"
            fi
        elif [[ -n "$SYS_PY" ]]; then
            echo "  Creating ${VENV_PATH} with ${SYS_PY} -m venv..."
            "$SYS_PY" -m venv "$VENV_PATH"
        else
            echo "Error: Neither uv nor python3 is available." >&2
            exit 1
        fi
        ok "Created ${VENV_PATH} ($("$VENV_PY" --version))"
    fi
fi

# 4. Python Packages & CUDA PyTorch
step "4/6 Checking CUDA PyTorch & MLIP/pytest Dependencies"
if [[ -x "$VENV_PY" ]]; then
    PROBE_JSON="$("$VENV_PY" - <<'PY'
import importlib.util, json
pkgs = [
    ("numpy", "numpy>=1.26"),
    ("scipy", "scipy>=1.11"),
    ("ase", "ase>=3.22"),
    ("mace", "mace-torch>=0.3.10"),
    ("torch_dftd", "torch-dftd>=0.5.0"),
    ("e3nn", "e3nn>=0.4.4"),
    ("matscipy", "matscipy>=1.1"),
    ("matplotlib", "matplotlib>=3.8"),
    ("pytest", "pytest>=8.0"),
]
missing = [spec for mod, spec in pkgs if importlib.util.find_spec(mod) is None]
torch_ok = importlib.util.find_spec("torch") is not None
torch_cuda = False
torch_ver = ""
torch_cuda_ver = ""
if torch_ok:
    import torch
    torch_ver = torch.__version__
    torch_cuda_ver = str(torch.version.cuda or "")
    torch_cuda = bool(torch.cuda.is_available())
print(json.dumps({
    "torch_ok": torch_ok,
    "torch_cuda": torch_cuda,
    "torch_ver": torch_ver,
    "torch_cuda_ver": torch_cuda_ver,
    "missing": missing,
}))
PY
)"
    TORCH_OK="$("$VENV_PY" -c "import json,sys; d=json.loads(sys.argv[1]); print(int(d['torch_ok'] and d['torch_cuda']))" "$PROBE_JSON")"
    TORCH_DESC="$("$VENV_PY" -c "import json,sys; d=json.loads(sys.argv[1]); print(f\"{d['torch_ver']} (cuda={d['torch_cuda_ver']}, available={d['torch_cuda']})\")" "$PROBE_JSON")"
    mapfile -t MISSING_PKGS < <("$VENV_PY" -c "import json,sys; d=json.loads(sys.argv[1]); print('\n'.join(d['missing']))" "$PROBE_JSON")

    if [[ "$TORCH_OK" -eq 1 ]]; then
        ok "PyTorch ${TORCH_DESC}"
    else
        miss "CUDA PyTorch missing or unusable (${TORCH_DESC})."
        if [[ "$CHECK_ONLY" -eq 1 ]]; then
            MISSING_ITEMS+=("CUDA PyTorch (${CUDA_INDEX}) in ${VENV_DIR}")
        else
            echo "  Installing CUDA PyTorch from ${TORCH_INDEX_URL} ..."
            if [[ -n "$UV_BIN" ]]; then
                "$UV_BIN" pip uninstall --python "$VENV_PY" torch >/dev/null 2>&1 || true
                env -u UV_DEFAULT_INDEX "$UV_BIN" pip install --python "$VENV_PY" --default-index "${TORCH_INDEX_URL}" torch
            else
                "$VENV_PY" -m pip uninstall -y torch >/dev/null 2>&1 || true
                "$VENV_PY" -m pip install --index-url "${TORCH_INDEX_URL}" torch
            fi
            "$VENV_PY" -c "import torch; assert torch.cuda.is_available(), f'Installed torch {torch.__version__} (cuda={torch.version.cuda}) cannot initialize CUDA on this driver'"
            ok "Installed CUDA PyTorch (${CUDA_INDEX})"
        fi
    fi

    # Filter out possible empty line from mapfile when missing list is empty
    CLEAN_MISSING=()
    for p in "${MISSING_PKGS[@]}"; do
        [[ -n "$p" ]] && CLEAN_MISSING+=("$p")
    done

    if [[ "${#CLEAN_MISSING[@]}" -eq 0 ]]; then
        ok "All MLIP & pytest Python packages present."
    else
        miss "Missing Python packages: ${CLEAN_MISSING[*]}"
        if [[ "$CHECK_ONLY" -eq 1 ]]; then
            MISSING_ITEMS+=("Python packages: ${CLEAN_MISSING[*]}")
        else
            INSTALLED_TORCH="$("$VENV_PY" -c "import torch; print(torch.__version__)")"
            CONSTRAINT_FILE="$(mktemp)"
            trap 'rm -f "$CONSTRAINT_FILE"' EXIT
            printf 'torch==%s\n' "$INSTALLED_TORCH" > "$CONSTRAINT_FILE"
            echo "  Installing missing packages (pinning torch==${INSTALLED_TORCH})..."
            if [[ -n "$UV_BIN" ]]; then
                "$UV_BIN" pip install --python "$VENV_PY" \
                    --extra-index-url "${TORCH_INDEX_URL}" \
                    --index-strategy unsafe-best-match \
                    -c "$CONSTRAINT_FILE" \
                    "${CLEAN_MISSING[@]}"
            else
                "$VENV_PY" -m pip install \
                    --extra-index-url "${TORCH_INDEX_URL}" \
                    -c "$CONSTRAINT_FILE" \
                    "${CLEAN_MISSING[@]}"
            fi
            rm -f "$CONSTRAINT_FILE"
            trap - EXIT
            "$VENV_PY" -c "import torch; assert torch.cuda.is_available(), f'CUDA unavailable after install: {torch.__version__}'; print('  [OK] Verified torch', torch.__version__, 'CUDA:', torch.version.cuda, 'GPU:', torch.cuda.get_device_name(0))"
        fi
    fi
fi

# 5. Build Core CUDA Binaries
step "5/6 Checking CUDA Binaries (build/mof_gcmc_cli & build/libmof_gcmc_proposer.so)"
CLI_BIN="$ROOT/build/mof_gcmc_cli"
PROP_LIB="$ROOT/build/libmof_gcmc_proposer.so"
NEED_BUILD=0
if [[ "$REBUILD" -eq 1 || ! -x "$CLI_BIN" || ! -f "$PROP_LIB" ]]; then
    NEED_BUILD=1
else
    if find "$ROOT/include" "$ROOT/src" -maxdepth 2 -type f \( -newer "$CLI_BIN" -o -newer "$PROP_LIB" \) | grep -q .; then
        echo "  Source files are newer than existing binaries; rebuild required."
        NEED_BUILD=1
    fi
fi

if [[ "$NEED_BUILD" -eq 0 ]]; then
    ok "Core binaries are up-to-date."
elif [[ "$CHECK_ONLY" -eq 1 || "$SKIP_BUILD" -eq 1 ]]; then
    if [[ "$CHECK_ONLY" -eq 1 ]]; then
        miss "Core binaries missing or outdated (run './fast_start.sh' to build)."
        MISSING_ITEMS+=("Core CUDA build")
    else
        warn "Skipping build (--skip-build)."
    fi
else
    bash "$ROOT/build.sh" core
    ok "Built mof_gcmc_cli and libmof_gcmc_proposer.so (${CUDA_ARCH})"
fi

if [[ "$CHECK_ONLY" -eq 1 ]]; then
    step "CheckOnly Summary"
    if [[ "${#MISSING_ITEMS[@]}" -eq 0 ]]; then
        ok "All Linux prerequisites and builds are ready!"
        exit 0
    else
        miss "Missing items: ${MISSING_ITEMS[*]}"
        exit 2
    fi
fi

# 6. Verification & Quickstart Commands
if [[ "$SKIP_TEST" -eq 0 ]]; then
    step "6/6 Running Self-Checks (CUDA_VISIBLE_DEVICES=${SEL_GPU})"
    echo "  [a] Running Pol-SCF unit verification (./build/mof_gcmc_cli --test-scf)..."
    "$CLI_BIN" --test-scf
    ok "mof_gcmc_cli --test-scf passed."

    if [[ "$FULL_TEST" -eq 1 ]]; then
        PYTEST_EXPR="not mlip"
    else
        PYTEST_EXPR="not mlip and not slow"
    fi
    echo "  [b] Running pytest regression suite (-m '${PYTEST_EXPR}')..."
    "$VENV_PY" -m pytest -v -m "${PYTEST_EXPR}"
    ok "Pytest checks (${PYTEST_EXPR}) passed."

    if [[ "$SMOKE_MLIP" -eq 1 ]]; then
        echo "  [c] Running live MLIP smoke test (IRMOF-1 + CH4, auto_mlip + MACE small)..."
        TMP_SMOKE="$ROOT/build/_tmp_faststart_smoke.json"
        rm -f "$TMP_SMOKE"
        set +e
        "$VENV_PY" -u "$ROOT/scripts/run_mlip_gcmc.py" \
            --cif "$ROOT/data/IRMOF-1.cif" --supercell 1 1 1 --temp 298 --pressures 5.0 \
            --guest ch4 --proposer auto_mlip --cbmc-k 128 --charge-mode auto \
            --model small --dispersion off --distill-samples-1b 160 --distill-samples-2b 32 \
            --boxes 2 --k-steps 16 --pilot-cycles 0 --eq-cycles 2 --prod-cycles 4 \
            --out-json "$TMP_SMOKE"
        SMOKE_RC=$?
        set -e
        rm -f "$TMP_SMOKE"
        if [[ "$SMOKE_RC" -ne 0 ]]; then
            echo "Error: Live MLIP smoke test failed with code ${SMOKE_RC}." >&2
            exit "$SMOKE_RC"
        fi
        ok "Live MLIP smoke test passed."
    fi
fi

printf '\n=== Fast-Start Complete! Ready on Linux (CUDA_VISIBLE_DEVICES=%s, %s) ===\n\n' "$SEL_GPU" "$CUDA_ARCH"
cat <<EOF
Copy-pasteable Linux runs:

1) Classical / Pol-SCF GPU GCMC (seconds):
   CUDA_VISIBLE_DEVICES=${SEL_GPU} ./build/mof_gcmc_cli --config configs/mg_mof74_co2_298K_uff.json

2) Online-distilled auto_mlip Nested MC (Mg-MOF-74 + CO2, r2SCAN):
   CUDA_VISIBLE_DEVICES=${SEL_GPU} ${VENV_DIR}/bin/python -u scripts/run_mlip_gcmc.py \\
     --cif data/MgMOF-74.cif --supercell 1 1 2 --temp 298 --pressures 0.012 0.28 1.0 \\
     --guest co2 --proposer auto_mlip --cbmc-k 128 --model mace-matpes-r2scan-0 --dispersion off \\
     --boxes 4 --k-steps 16 --pilot-cycles 40 --eq-cycles 1200 --prod-cycles 4800 \\
     --out-json build/mgmof74_co2_r2scan.json --out-csv build/mgmof74_co2_r2scan.csv

3) Full classical parity or full pytest suite:
   ./fast_start.sh --full-test
   CUDA_VISIBLE_DEVICES=${SEL_GPU} ${VENV_DIR}/bin/python -m pytest -v
EOF
