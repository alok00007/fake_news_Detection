import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("hamzab/roberta-fake-news-classification")
model = AutoModelForSequenceClassification.from_pretrained("hamzab/roberta-fake-news-classification")

# Fake News Detection Function
def detect_fake_news(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=-1).item()
    return "Fake News" if prediction == 0 else "Real News"

# Streamlit App
st.title("Fake News Detection")
st.write("Enter a news headline or text to check if it's real or fake.")

user_input = st.text_area("Enter the news text:", "")

if st.button("Analyze"):
    if user_input.strip():
        result = detect_fake_news(user_input)
        st.write(f"Result: **{result}**")
    else:
        st.write("Please enter some text to analyze.")
