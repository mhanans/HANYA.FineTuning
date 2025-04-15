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
    # Instal torch dari indeks khusus untuk CPU
    pip install torch --index-url https://download.pytorch.org/whl/cpu || {
        echo "Gagal menginstal torch."
        exit 1
    }
    # Instal dependensi lainnya dari PyPI default
    pip install transformers datasets peft gradio pdfplumber ollama || {
        echo "Gagal menginstal dependensi lainnya."
        exit 1
    }
}

# Fungsi untuk memeriksa dan menjalankan pembuatan .jsonl dari PDF
generate_jsonl() {
    echo "Memeriksa file PDF di direktori $PDF_DIR..."
    if [ -d "$PDF_DIR" ] && [ -n "$(ls -A "$PDF_DIR"/*.pdf 2>/dev/null)" ]; then
        echo "File PDF ditemukan. Menjalankan generate_jsonl.py..."
        python generate_jsonl.py || {
            echo "Gagal menjalankan generate_jsonl.py."
            exit 1
        }
    else
        echo "Tidak ada file PDF di $PDF_DIR. Melewati langkah pembuatan .jsonl."
    fi
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
INSTALL_DIR="$(pwd)/install_dir"  # Direktori untuk instalasi
CONDA_ROOT="$INSTALL_DIR/conda"   # Lokasi Miniconda
ENV_DIR="$INSTALL_DIR/env"        # Lokasi environment Conda
PYTHON_VERSION="3.10"             # Versi Python yang digunakan
PDF_DIR="./pdfs"                  # Direktori untuk file PDF

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
echo "Membuat file .jsonl dari PDF (jika ada)"
echo "******************************************************"
generate_jsonl

echo "******************************************************"
echo "Menjalankan fine-tuning"
echo "******************************************************"
run_finetune

echo "******************************************************"
echo "Menjalankan antarmuka Gradio"
echo "******************************************************"
run_gradio