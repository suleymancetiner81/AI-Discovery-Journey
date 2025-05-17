import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Model ve tokenizer'ı yükle
@st.cache_resource
def load_model():
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

st.title("🧠 Basit GPT-2 Metin Tamamlama")

prompt = st.text_area("Metin başlangıcını girin:", "Bir zamanlar uzak bir köyde")

def generate_text(prompt, max_length=100):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(
        input_ids,
        max_length=max_length,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        top_k=50,
        top_p=0.95,
        temperature=0.7,
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)

if st.button("Tamamla"):
    st.info("Metin oluşturuluyor...")
    completed_text = generate_text(prompt)
    st.subheader("Tamamlanan Metin:")
    st.write(completed_text)