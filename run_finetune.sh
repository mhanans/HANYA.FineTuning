#!/bin/bash

# Fungsi untuk menginstal Miniconda
install_miniconda() {
    local sys_arch=$(uname -m)
    case "${sys_arch}" in
    x86_64*) sys_arch="x86_64" ;;
    arm64*|aarch64*) sys_arch="aarch64" ;;
    *) echo "Arsitektur sistem tidak didukung: ${sys_arch}"; exit 1 ;;
    esac

    local miniconda_url="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-${sys_arch}.sh"
    if ! "${CONDA_ROOT}/bin/conda" --version &>/dev/null; then
        mkdir -p "$INSTALL_DIR"
        curl -Lk "$miniconda_url" >"$INSTALL_DIR/miniconda_installer.sh"
        chmod u+x "$INSTALL_DIR/miniconda_installer.sh"
        bash "$INSTALL_DIR/miniconda_installer.sh" -b -p "$CONDA_ROOT"
        rm -rf "$INSTALL_DIR/miniconda_installer.sh"
    fi
    echo "Miniconda terinstal di $CONDA_ROOT"
}

# Fungsi untuk membuat environment Conda
create_conda_env() {
    if [ ! -d "$ENV_DIR" ]; then
        echo "Membuat environment Conda dengan Python $PYTHON_VERSION di $ENV_DIR"
        "${CONDA_ROOT}/bin/conda" create -y -k --prefix "$ENV_DIR" python="$PYTHON_VERSION" || {
            echo "Gagal membuat environment Conda."
            rm -rf "$ENV_DIR"
            exit 1
        }
    else
        echo "Environment Conda sudah ada di $ENV_DIR"
    fi
}

# Fungsi untuk mengaktifkan environment Conda
activate_conda_env() {
    source "$CONDA_ROOT/etc/profile.d/conda.sh"
    conda activate "$ENV_DIR" || {
        echo "Gagal mengaktifkan environment. Hapus $ENV_DIR dan jalankan ulang skrip."
        exit 1
    }
    echo "Environment Conda diaktifkan di $CONDA_PREFIX"
}

# Fungsi untuk menginstal dependensi
install_dependencies() {
    echo "Menginstal dependensi..."
    pip install torch --index-url https://download.pytorch.org/whl/cpu transformers datasets peft gradio || {
        echo "Gagal menginstal dependensi."
        exit 1
    }
}

# Fungsi untuk menjalankan fine-tuning
run_finetune() {
    echo "Menjalankan fine-tuning..."
    python finetune.py || {
        echo "Gagal menjalankan fine-tuning."
        exit 1
    }
}

# Fungsi untuk menjalankan Gradio
run_gradio() {
    echo "Menjalankan antarmuka Gradio..."
    python gradio_app.py || {
        echo "Gagal menjalankan Gradio."
        exit 1
    }
}

# Eksekusi utama
INSTALL_DIR="$(pwd)/install_dir"
CONDA_ROOT="$INSTALL_DIR/conda"
ENV_DIR="$INSTALL_DIR/env"
PYTHON_VERSION="3.10"

echo "******************************************************"
echo "Menyiapkan Miniconda"
echo "******************************************************"
install_miniconda

echo "******************************************************"
echo "Membuat environment Conda"
echo "******************************************************"
create_conda_env
activate_conda_env

echo "******************************************************"
echo "Menginstal dependensi"
echo "******************************************************"
install_dependencies

echo "******************************************************"
echo "Menjalankan fine-tuning"
echo "******************************************************"
run_finetune

echo "******************************************************"
echo "Menjalankan antarmuka Gradio"
echo "******************************************************"
run_gradio