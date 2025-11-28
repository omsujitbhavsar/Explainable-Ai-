import streamlit as st
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from lime.lime_text import LimeTextExplainer
import hashlib
import base64
import os
import requests
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend

# --- Setup ---
st.set_page_config(page_title="Offensive Comment Classifier", layout="wide")
st.title("🚨 Offensive Comment Classifier")
st.write("Upload a CSV file or enter text to classify comments as offensive/non-offensive.")


# --- Model Loading ---
@st.cache_resource
def load_model():
    model_path = "/Users/ombhavsar/PycharmProjects/Explainable_Ai/trained_model"
    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        return model, tokenizer
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return None, None


model, tokenizer = load_model()
if model is None:
    st.stop()

model.eval()
explainer = LimeTextExplainer(class_names=["Non-offensive", "Mildly offensive", "Highly offensive"])

# --- Security Setup ---
PINATA_API_KEY = 'b3ccb860124de275a20a'
PINATA_SECRET = '5abd3c0045ec8bef2d7ac64b7d1bb5842df9ac4340df82b8b3b477babd3febfe'
secret_key = os.urandom(32)  # AES-256


# --- Encryption Functions ---
def encrypt_data(data, key):
    iv = os.urandom(16)
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    encryptor = cipher.encryptor()
    pad_len = 16 - len(data) % 16
    data_padded = data + chr(pad_len) * pad_len
    encrypted = encryptor.update(data_padded.encode()) + encryptor.finalize()
    return base64.b64encode(encrypted).decode(), base64.b64encode(iv).decode()


def generate_hash(data):
    return hashlib.sha256(data.encode()).hexdigest()


# --- IPFS Functions ---
def upload_to_ipfs(content):
    headers = {
        "pinata_api_key": PINATA_API_KEY,
        "pinata_secret_api_key": PINATA_SECRET,
    }
    try:
        response = requests.post(
            "https://api.pinata.cloud/pinning/pinFileToIPFS",
            files={"file": ("explanation.txt", content.encode())},
            headers=headers,
            timeout=10
        )
        response.raise_for_status()
        return response.json()["IpfsHash"]
    except Exception as e:
        st.error(f"IPFS upload failed: {str(e)}")
        return None


# --- Classification Functions ---
def classify_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=1)[0]
    pred_class = torch.argmax(probs).item()
    return pred_class, probs[pred_class].item()


def explain_text(text):
    def predictor(texts):
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.logits.detach().numpy()

    explanation = explainer.explain_instance(text, predictor, num_features=5)
    return explanation.as_list()


# --- Main Processing Function ---
def process_text(text):
    with st.spinner("Analyzing..."):
        # Classification
        class_map = {0: "Non-offensive", 1: "offensive"}
        pred_class, confidence = classify_text(text)

        result = {
            "text": text,
            "classification": class_map[pred_class],
            "confidence": confidence,
            "ipfs_cid": None,
            "explanation": None
        }

        # Only process explanations for offensive content
        if pred_class > 0:  # 1 or 2 (offensive)
            explanation = explain_text(text)
            explanation_text = "\n".join([f"{word}: {weight:.2f}" for word, weight in explanation])

            # Security processing
            explanation_hash = generate_hash(explanation_text)
            encrypted_data, iv = encrypt_data(explanation_text, secret_key)
            cid = upload_to_ipfs(encrypted_data)

            result.update({
                "explanation": explanation,
                "hash": explanation_hash,
                "ipfs_cid": cid,
                "iv": iv
            })

        return result


# --- UI Components ---
tab1, tab2 = st.tabs(["📁 Upload CSV", "✍️ Single Text"])

with tab1:
    st.header("Batch Processing (CSV)")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"], key="csv_uploader")

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            if "text" not in df.columns:
                st.error("CSV must contain a 'text' column")
            else:
                st.success(f"Loaded {len(df)} records")
                st.dataframe(df.head())

                if st.button("Process All", key="process_csv"):
                    results = []
                    progress_bar = st.progress(0)

                    for i, row in enumerate(df.itertuples()):
                        result = process_text(row.text)
                        results.append({
                            "text": result["text"],
                            "classification": result["classification"],
                            "confidence": result["confidence"],
                            "ipfs_cid": result["ipfs_cid"]
                        })
                        progress_bar.progress((i + 1) / len(df))

                    results_df = pd.DataFrame(results)
                    st.header("Results")
                    st.dataframe(results_df)
                    st.subheader("Explanation")

                    # Download button
                    st.download_button(
                        label="Download Results",
                        data=results_df.to_csv(index=False),
                        file_name="classification_results.csv",
                        mime="text/csv"
                    )

                    # Show detailed explanations only for offensive content
                    offensive_results = [r for r in results if r["ipfs_cid"] is not None]
                    if offensive_results:
                        st.header("Offensive Content Explanations")
                        for result in offensive_results:
                            with st.expander(f"Explanation for: {result['text']}"):
                                st.write(f"**Classification**: {result['classification']}")
                                st.write(f"**Confidence**: {result['confidence']:.2f}")
                                st.write(f"**IPFS CID**: `{result['ipfs_cid']}`")
                                st.write(f"View on IPFS: https://gateway.pinata.cloud/ipfs/{result['ipfs_cid']}")

        except Exception as e:
            st.error(f"Error processing CSV: {str(e)}")

with tab2:
    st.header("Single Text Analysis")
    input_text = st.text_area("Enter text to analyze:", height=100)

    if st.button("Analyze", key="analyze_single"):
        if not input_text.strip():
            st.warning("Please enter some text")
        else:
            result = process_text(input_text.strip())

            st.subheader("Results")
            col1, col2 = st.columns(2)

            with col1:
                st.metric("Classification", result["classification"])
                st.metric("Confidence", f"{result['confidence']:.2f}")

            with col2:
                if result["ipfs_cid"]:
                    st.metric("IPFS CID", result["ipfs_cid"])
                    st.markdown(f"[View on IPFS](https://gateway.pinata.cloud/ipfs/{result['ipfs_cid']})")

            if result["classification"] != "Non-offensive":
                st.subheader("Explanation")
                for word, weight in result["explanation"]:
                    st.write(f"- {word}: {weight:.2f}")

# --- Footer ---
st.markdown("---")
st.markdown("""
<style>
.footer {
    font-size: small;
    color: gray;
    text-align: center;
}
</style>
<div class="footer">
    Offensive content explanations are encrypted and stored on IPFS for transparency
</div>
""", unsafe_allow_html=True)
