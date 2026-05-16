import transformers as tr
from transformers import AutoTokenizer, AutoModelForCausalLM
import streamlit as st
import torch

model = 'facebook/bart-large-cnn'
token = AutoTokenizer.from_pretrained(model)
model = AutoModelForCausalLM.from_pretrained(model)
st.title("Class Summarizer")
st.write("Enter the class description and the class name to get a summary of the class.")
class_name = st.text_input("Class Name")
class_description = st.text_area("Class Description")
if st.button("Summarize"):
    input_text = f"Class Name: {class_name}\nClass Description: {class_description}\nSummary:"
    inputs = token(input_text, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=100)
    summary = token.decode(outputs[0], skip_special_tokens=True)
    st.write("Summary:")
    st.write(summary)