from textblob import TextBlob
from deep_translator import GoogleTranslator

text = input("Bir cümle yazın: ")

try:
    # Türkçeden İngilizceye çeviri
    translated_text = GoogleTranslator(source='auto', target='en').translate(text)
    blob = TextBlob(translated_text)

    polarity = blob.sentiment.polarity

    print(f"\nOrijinal: {text}")
    print(f"İngilizce: {translated_text}")
    print(f"Duygu değeri (polarity): {polarity}")

    if polarity > 0:
        print("Yorum: Bu cümle pozitif 🌞")
    elif polarity < 0:
        print("Yorum: Bu cümle negatif 🌧️")
    else:
        print("Yorum: Nötr bir ifade 😐")
except Exception as e:
    print("Hata oluştu:", e)