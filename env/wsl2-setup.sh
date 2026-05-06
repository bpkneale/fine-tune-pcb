#!/usr/bin/env bash
# Set up the WSL2 + ROCm + PyTorch + HF environment for fine-tune-pcb.
# Idempotent: every step checks if the work is already done and skips it.
#
# Manual prerequisites this script CANNOT do (Windows side):
#   - Install AMD Adrenalin 26.2.2 or later on the Windows host.
#   - WSL2 itself (`wsl --install -d Ubuntu-24.04`).
#   - After install, you may need to run `wsl --shutdown` from a Windows
#     PowerShell once and reopen WSL for the GPU passthrough to come up.
#
# Everything else — ROCm runtime, librocdxg, python venv, torch-rocm, HF
# stack — runs automatically. `sudo` will prompt once for your password.
#
# Usage:
#   bash env/wsl2-setup.sh
#
# Environment overrides:
#   ROCM_VERSION       — e.g. "7.2.3" (default: latest known good)
#   VENV               — venv path (default: $HOME/venvs/pcb)
#   SKIP_ROCM_INSTALL  — set to 1 to skip ROCm install even if rocminfo missing
#   SKIP_LIBROCDXG     — set to 1 to skip librocdxg build (only fine if you
#                         confirmed rocminfo already detects the GPU)

set -euo pipefail

ROCM_VERSION="${ROCM_VERSION:-7.2.3}"
VENV="${VENV:-$HOME/venvs/pcb}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

log()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m!!\033[0m  %s\n' "$*" >&2; }
fail() { printf '\033[1;31mxx\033[0m  %s\n' "$*" >&2; exit 1; }

# ----------------------------------------------------------------------------
# 0. Distro check
# ----------------------------------------------------------------------------
if [[ ! -f /etc/os-release ]]; then
    fail "/etc/os-release missing — can't detect distro"
fi
# shellcheck disable=SC1091
. /etc/os-release
case "${UBUNTU_CODENAME:-${VERSION_CODENAME:-}}" in
    noble)  CODENAME=noble  ;;   # 24.04
    jammy)  CODENAME=jammy  ;;   # 22.04
    *)
        warn "AMD only supports Ubuntu 24.04 (noble) or 22.04 (jammy) for ROCm in WSL."
        warn "Detected: ${PRETTY_NAME:-unknown}"
        warn "Continuing anyway — install will likely fail. Re-export CODENAME=noble to override."
        CODENAME="${CODENAME:-noble}"
        ;;
esac
log "Distro: ${PRETTY_NAME} (using $CODENAME repo)"

# ----------------------------------------------------------------------------
# 1. Sanity: WSL passthrough device
# ----------------------------------------------------------------------------
if [[ ! -e /dev/dxg ]]; then
    warn "/dev/dxg not present — Windows hasn't exposed the GPU to WSL yet."
    warn "Update AMD Adrenalin (>=26.2.2) on Windows, then run \`wsl --shutdown\` from PowerShell."
fi

# ----------------------------------------------------------------------------
# 2. Install ROCm (idempotent — skipped if rocminfo already works)
# ----------------------------------------------------------------------------
need_rocm=1
if [[ "${SKIP_ROCM_INSTALL:-0}" == "1" ]]; then
    need_rocm=0
elif command -v rocminfo >/dev/null 2>&1 && rocminfo >/dev/null 2>&1; then
    log "rocminfo already works — skipping ROCm install"
    need_rocm=0
fi

if [[ "$need_rocm" == "1" ]]; then
    log "Installing ROCm $ROCM_VERSION via amdgpu-install"
    sudo apt-get update -y
    sudo apt-get install -y wget gnupg ca-certificates

    # Filename pattern: amdgpu-install_X.Y.Z.AAABCC-1_all.deb
    # The "AAABCC" suffix follows AMD's release tag — match by glob after download.
    deb_url="https://repo.radeon.com/amdgpu-install/${ROCM_VERSION}/ubuntu/${CODENAME}/"
    log "Discovering .deb under $deb_url"
    deb_name="$(curl -fsSL "$deb_url" | grep -oE 'amdgpu-install_[0-9.]+-1_all\.deb' | head -1 || true)"
    if [[ -z "$deb_name" ]]; then
        # Construct from version: 7.2.3 -> 7.2.3.70203-1
        # Best-effort fallback; user can override ROCM_VERSION.
        fail "Could not find amdgpu-install .deb at $deb_url
       Check that ROCM_VERSION=$ROCM_VERSION is correct for $CODENAME."
    fi
    log "Downloading $deb_name"
    cd /tmp
    wget -q --show-progress "${deb_url}${deb_name}" -O "$deb_name"

    log "Installing $deb_name"
    sudo apt-get install -y "./$deb_name"
    sudo apt-get update -y

    log "Running amdgpu-install --usecase=rocm --no-dkms (this is the long step)"
    # The kernel driver lives on the Windows side, so --no-dkms skips the
    # kernel module build that would otherwise fail in WSL. The 'wsl'
    # usecase mentioned in some AMD docs does not exist in this installer.
    sudo amdgpu-install -y --usecase=rocm --no-dkms

    log "Adding $USER to render and video groups"
    sudo usermod -a -G render,video "$USER"

    cd "$REPO_ROOT"
fi

# ----------------------------------------------------------------------------
# 3. librocdxg (the WSL bridge between ROCm and /dev/dxg)
# ----------------------------------------------------------------------------
need_librocdxg=1
if [[ "${SKIP_LIBROCDXG:-0}" == "1" ]]; then
    need_librocdxg=0
elif rocminfo 2>/dev/null | grep -q "Marketing Name"; then
    log "rocminfo lists devices already — skipping librocdxg build"
    need_librocdxg=0
fi

if [[ "$need_librocdxg" == "1" ]]; then
    log "Building librocdxg"
    sudo apt-get install -y git build-essential cmake pkg-config

    # Discover newest installed Windows SDK version on the Windows side.
    WIN_SDK_INCLUDE='/mnt/c/Program Files (x86)/Windows Kits/10/Include'
    if [[ ! -d "$WIN_SDK_INCLUDE" ]]; then
        fail "Windows SDK not found at $WIN_SDK_INCLUDE
       Install one (e.g. via Visual Studio Build Tools) and rerun."
    fi
    sdk_version="$(ls "$WIN_SDK_INCLUDE" | grep -E '^10\.' | sort -V | tail -1 || true)"
    if [[ -z "$sdk_version" ]]; then
        fail "Could not find a 10.* version under $WIN_SDK_INCLUDE"
    fi
    log "Using Windows SDK $sdk_version"
    win_sdk="$WIN_SDK_INCLUDE/$sdk_version/shared"

    src_dir="$HOME/src/librocdxg"
    if [[ ! -d "$src_dir" ]]; then
        git clone https://github.com/ROCm/librocdxg.git "$src_dir"
    else
        ( cd "$src_dir" && git pull --ff-only )
    fi
    mkdir -p "$src_dir/build"
    cd "$src_dir/build"
    cmake .. -DWIN_SDK="$win_sdk"
    make -j"$(nproc)"
    sudo make install

    cd "$REPO_ROOT"

    # Make HSA_ENABLE_DXG_DETECTION persistent for this user
    if ! grep -q HSA_ENABLE_DXG_DETECTION "$HOME/.bashrc" 2>/dev/null; then
        cat >> "$HOME/.bashrc" <<'EOF'

# Required for ROCm to discover the WSL GPU passthrough
export HSA_ENABLE_DXG_DETECTION=1
EOF
    fi
fi
export HSA_ENABLE_DXG_DETECTION=1

# ----------------------------------------------------------------------------
# 4. Python toolchain
# ----------------------------------------------------------------------------
log "Installing python toolchain"
if command -v python3.12 >/dev/null 2>&1; then
    PYTHON=python3.12
    sudo apt-get install -y python3.12 python3.12-venv python3-pip
elif command -v python3.11 >/dev/null 2>&1; then
    PYTHON=python3.11
    sudo apt-get install -y python3.11 python3.11-venv python3-pip
else
    PYTHON=python3
    sudo apt-get install -y python3 python3-venv python3-pip
fi
log "Interpreter: $($PYTHON --version)"

# ----------------------------------------------------------------------------
# 5. Project venv
# ----------------------------------------------------------------------------
if [[ ! -d "$VENV" ]]; then
    log "Creating venv at $VENV"
    mkdir -p "$(dirname "$VENV")"
    "$PYTHON" -m venv "$VENV"
else
    log "Reusing venv at $VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install --upgrade pip wheel

# ----------------------------------------------------------------------------
# 6. PyTorch + HF stack
# ----------------------------------------------------------------------------
# For ROCm 7.x we install AMD's prebuilt wheels from repo.radeon.com — the
# stable wheels at download.pytorch.org/whl/rocmX.Y lag and are built against
# ROCm <=6.3, which fails to enumerate devices on a ROCm 7.2.3 system.
# For ROCm 6.x we use the pytorch.org wheels.
case "$ROCM_VERSION" in
    7.*)
        # Discover the matching wheel filenames live so the version pins
        # don't go stale when AMD ships a new minor.
        AMD_INDEX="https://repo.radeon.com/rocm/manylinux/rocm-rel-${ROCM_VERSION}/"
        log "Discovering AMD torch wheels at $AMD_INDEX"
        listing="$(curl -fsSL "$AMD_INDEX" || true)"
        if [[ -z "$listing" ]]; then
            fail "Could not list $AMD_INDEX — check that ROCM_VERSION=$ROCM_VERSION matches a published directory."
        fi
        pick() {
            local pkg="$1"
            echo "$listing" \
                | grep -oE "${pkg}-[^\"]*cp312[^\"]*\.whl" \
                | sort -V | tail -1
        }
        triton_whl="$(pick triton)"
        torch_whl="$(pick torch)"
        tv_whl="$(pick torchvision)"
        ta_whl="$(pick torchaudio)"
        for v in triton_whl torch_whl tv_whl ta_whl; do
            [[ -n "${!v}" ]] || fail "missing wheel: $v"
        done
        log "Installing AMD-built torch stack for ROCm $ROCM_VERSION"
        pip install --upgrade "${AMD_INDEX}${triton_whl}"
        pip install --upgrade \
            "${AMD_INDEX}${torch_whl}" \
            "${AMD_INDEX}${tv_whl}" \
            "${AMD_INDEX}${ta_whl}"
        ;;
    *)
        case "$ROCM_VERSION" in
            6.4*|6.3*) ROCM_TORCH=rocm6.3 ;;
            6.2*)      ROCM_TORCH=rocm6.2 ;;
            *)         ROCM_TORCH=rocm6.2 ;;
        esac
        log "Installing torch from pytorch.org (index: $ROCM_TORCH)"
        pip install --upgrade torch torchvision torchaudio \
            --index-url "https://download.pytorch.org/whl/${ROCM_TORCH}"
        ;;
esac

log "Installing HF training stack"
pip install --upgrade \
    transformers peft datasets accelerate trl \
    sentencepiece protobuf orjson tqdm

log "bitsandbytes / QLoRA: AMD doesn't publish a ROCm 7.x wheel; skipping."
log "  (train_lora.py automatically falls back to bf16 LoRA, which fits a 4B model in 24 GB.)"

# ----------------------------------------------------------------------------
# 7. Project install
# ----------------------------------------------------------------------------
log "Installing fine-tune-pcb (editable) from $REPO_ROOT"
pip install -e "$REPO_ROOT"

# ----------------------------------------------------------------------------
# 8. Sanity check
# ----------------------------------------------------------------------------
log "Verifying torch sees the GPU"
python <<'PY' || warn "torch GPU check failed — see message above."
import torch
print(f"torch         : {torch.__version__}")
print(f"cuda.is_avail : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"device        : {torch.cuda.get_device_name(0)}")
    print(f"device_count  : {torch.cuda.device_count()}")
else:
    raise SystemExit("torch.cuda.is_available() is False")
PY

# ----------------------------------------------------------------------------
# 9. Manual finishing steps
# ----------------------------------------------------------------------------
cat <<EOF

\033[1;32m==> Setup complete.\033[0m

If you just installed ROCm/librocdxg for the first time, log out and back
into WSL once so the render/video group memberships take effect:

       exit        # then reopen WSL

Then two interactive steps:

  1. Hugging Face auth (Gemma is a gated model):

       source $VENV/bin/activate
       huggingface-cli login
       # paste a token (read scope) from https://huggingface.co/settings/tokens

  2. Accept Gemma 4 terms once in a browser:

       https://huggingface.co/google/gemma-4-E4B-it

Then start the smoke run:

       cd $REPO_ROOT
       source $VENV/bin/activate
       bash scripts/e2e_smoke.sh

EOF
