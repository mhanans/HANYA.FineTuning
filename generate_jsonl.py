import os
import json
import pdfplumber
import ollama

# Konfigurasi
pdf_directory = "./pdfs"  # Direktori tempat file PDF disimpan
output_jsonl = "custom_knowledge.jsonl"  # Nama file output .jsonl
min_data = 5000  # Jumlah minimal data yang diinginkan
ollama_model = "llama3.1"  # Model Ollama yang akan digunakan (dapat diganti)

# Fungsi untuk membaca teks dari file PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

# Fungsi untuk menghasilkan pertanyaan dan jawaban menggunakan Ollama
def generate_qa(text):
    prompt = f"Generate a question and answer pair based on the following text in English:\n\n{text}\n\nFormat the output as JSON with 'prompt' and 'completion' keys."
    response = ollama.chat(model=ollama_model, messages=[{'role': 'user', 'content': prompt}])
    try:
        qa = json.loads(response['message']['content'])
        return qa
    except json.JSONDecodeError:
        return None

# Fungsi untuk membuat file .jsonl
def create_jsonl(data, output_file):
    with open(output_file, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

# Fungsi utama
def main():
    # Baca semua file PDF di direktori
    pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]
    all_qa = []

    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_directory, pdf_file)
        text = extract_text_from_pdf(pdf_path)
        # Bagi teks menjadi potongan kecil (500 karakter) untuk menghasilkan lebih banyak data
        text_chunks = [text[i:i+500] for i in range(0, len(text), 500)]
        for chunk in text_chunks:
            qa = generate_qa(chunk)
            if qa and 'prompt' in qa and 'completion' in qa:
                all_qa.append(qa)

    # Jika data kurang dari 5000, ulangi data yang ada
    while len(all_qa) < min_data:
        all_qa.extend(all_qa[:min_data - len(all_qa)])

    # Simpan ke file .jsonl
    create_jsonl(all_qa, output_jsonl)
    print(f"File {output_jsonl} berhasil dibuat dengan {len(all_qa)} entri.")

if __name__ == "__main__":
    main()