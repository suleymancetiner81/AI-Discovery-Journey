import os
from gtts import gTTS
from PyPDF2 import PdfReader

def pdf_to_text(pdf_path):
    """PDF dosyasından metni çıkarır."""
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text

def text_to_audio(text, output_path="output.mp3", lang="tr"):
    """Metni ses dosyasına dönüştürür."""
    tts = gTTS(text=text, lang=lang, slow=False)
    tts.save(output_path)
    print(f"Ses dosyası oluşturuldu: {output_path}")

if __name__ == "__main__":
    pdf_path = input("PDF dosyasının yolunu girin: ").strip('"')
    
    if not os.path.exists(pdf_path):
        print("Hata: PDF dosyası bulunamadı!")
    else:
        text = pdf_to_text(pdf_path)
        if not text.strip():
            print("Hata: PDF'den metin çıkarılamadı (tarama yapılmış PDF olabilir).")
        else:
            text_to_audio(text, lang="tr")  # Türkçe için 'tr', İngilizce için 'en'