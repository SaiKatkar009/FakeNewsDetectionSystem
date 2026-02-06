"""
app_improved.py — Clean UI + Explanation + NO temperature scaling
"""

import os
import re
import json
import numpy as np
import streamlit as st

# Try TensorFlow first
USE_TF = True
try:
    import tensorflow as tf
    from transformers import DistilBertTokenizerFast, TFDistilBertForSequenceClassification
except Exception:
    USE_TF = False

# Try PyTorch
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False


# ----------------------- CLEAN TEXT -----------------------
def clean_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


# --------------------- TF GRADIENT EXPLANATION ---------------------
def tf_token_importances(tf_model, tokenizer, text):
    try:
        inputs = tokenizer(text, return_tensors="tf", truncation=True, padding=True, max_length=256)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        with tf.GradientTape() as tape:
            embeddings = tf_model.distilbert.embeddings(input_ids)
            tape.watch(embeddings)
            outputs = tf_model(input_ids=input_ids, attention_mask=attention_mask, training=False)
            logits = outputs.logits[:, 1]   # logit for label=1 (real)

        grads = tape.gradient(logits, embeddings)
        if grads is None:
            return None

        scores = tf.reduce_mean(tf.abs(grads * embeddings), axis=-1).numpy().squeeze()
        tokens = tokenizer.convert_ids_to_tokens(input_ids.numpy().squeeze().tolist())

        filtered = [
            (t, float(s)) for t, s in zip(tokens, scores)
            if t not in ["[PAD]", "[CLS]", "[SEP]"]
        ]

        total = sum(abs(s) for _, s in filtered) + 1e-12
        filtered_norm = [(t, s / total) for t, s in filtered]
        return filtered_norm

    except Exception:
        return None


# ------------------ TORCH ATTENTION EXPLANATION -------------------
def torch_attention_importances(pt_model, tokenizer, text):
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pt_model.to(device)
        pt_model.eval()

        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256).to(device)

        with torch.no_grad():
            outputs = pt_model(**inputs, output_attentions=True, return_dict=True)

        attentions = outputs.attentions
        att_mat = torch.stack(attentions, dim=0)
        att_mean = att_mat.mean(dim=0).mean(dim=1)
        att_scores = att_mean.sum(dim=1).squeeze().cpu().numpy()

        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze().cpu().tolist())
        filtered = [
            (t, float(s)) for t, s in zip(tokens, att_scores)
            if t not in ["[PAD]", "[CLS]", "[SEP]"]
        ]

        total = sum(abs(s) for _, s in filtered) + 1e-12
        filtered_norm = [(t, s / total) for t, s in filtered]
        return filtered_norm
    except Exception:
        return None


# ------------------ BUILD SIMPLE REASON --------------------------
def build_reason_from_token_scores(token_scores, top_k=6):
    if not token_scores:
        return "No token-level explanation available."
    top = sorted(token_scores, key=lambda x: x[1], reverse=True)[:top_k]
    parts = [f"'{t}': {s:.3f}" for t, s in top]
    return "Top contributing tokens → " + ", ".join(parts)


# ------------------------ UI SETUP -------------------------
st.set_page_config(
    page_title="Fake News Detection — BERT",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🧠 Fake News Detection ")


# ---------------------- SIDEBAR ----------------------
with st.sidebar:
    st.header("Model Settings")
    model_dir = st.text_input("Model directory", value="model/bert")

    st.markdown("### Confidence threshold")
    threshold = st.slider("Threshold", 0.5, 0.99, 0.90, 0.01)

    st.markdown("---")
    st.markdown("### Example News Articles")

    examples = [
    (
        "Real — ISRO Satellite Mission",
        """The Indian Space Research Organisation (ISRO) has successfully launched its 
        latest Earth-observation satellite, EOS-09, from the Satish Dhawan Space Centre 
        in Sriharikota. The satellite, carried onboard the PSLV-C65 rocket, will monitor 
        coastal regions, agricultural patterns, and climate variations across the Indian 
        subcontinent. Scientists said the mission would significantly improve India’s 
        disaster-management capabilities by providing high-resolution images in near 
        real time. Prime Minister Modi congratulated the ISRO team for the achievement, 
        calling it another step toward strengthening India’s space infrastructure."""
    ),

    (
        "Real — RBI Policy Update",
        """The Reserve Bank of India (RBI) announced in its latest monetary policy 
        review that the repo rate will remain unchanged for the next quarter. The 
        Monetary Policy Committee stated that while inflation has eased, global 
        economic uncertainty and fluctuating crude oil prices require a cautious 
        approach. The central bank highlighted that India’s GDP growth has shown 
        resilience driven by strong consumer demand and stable manufacturing output. 
        Economists believe the RBI’s decision aims to maintain financial stability 
        while supporting ongoing economic recovery."""
    ),

    (
        "Fake — National Petrol Ban",
        """A series of online posts claim that the Indian government has approved a 
        nationwide ban on all petrol and diesel vehicles starting from 2027, replacing 
        them entirely with electric vehicles. According to the circulating reports, 
        citizens will be required to surrender their existing vehicles in exchange 
        for government-issued ‘mobility tokens’. These reports further claim that 
        fuel stations will be gradually shut down over the next three years. However, 
        the Ministry of Road Transport has issued a statement confirming that no such 
        policy exists and that the government has not proposed any plan mandating 
        the discontinuation of petrol vehicles."""
    ),

    (
        "Fake — Miracle Cancer Treatment",
        """Several unofficial blogs are claiming that researchers at an unnamed European 
        institute have discovered a plant-based compound capable of completely curing 
        cancer within 72 hours, without chemotherapy or surgery. The articles state 
        that the compound, supposedly extracted from a rare Amazonian root, can ‘detect 
        and destroy’ all malignant cells in the body. Medical experts, however, have 
        categorically denied the existence of such treatment, clarifying that no peer-
        reviewed publications or clinical trials support these claims. Oncologists warn 
        that such misinformation can mislead patients and encourage unsafe decisions."""
    ),

    (
        "Real — Indian Railways Upgrade",
        """The Indian Railways has begun modernising over 200 stations under the 
        ‘Amrit Bharat’ initiative aimed at improving passenger facilities and 
        upgrading infrastructure. The project includes revamped waiting areas, 
        enhanced surveillance systems, energy-efficient lighting, and improved 
        accessibility for differently-abled passengers. According to officials, work 
        is progressing in multiple zones simultaneously, with completion expected 
        within two years. The initiative is part of a broader effort to enhance safety, 
        comfort, and operational efficiency across the national transport network."""
    ),

    (
        "Fake — Currency Replacement",
        """A fabricated news report circulating online claims that the Reserve Bank 
        of India will invalidate all existing ₹500 and ₹2000 notes and replace them 
        with new ‘digital-embedded currency’. The report states that citizens will 
        have only ten days to exchange their existing notes, after which they will 
        be considered illegal. The RBI has dismissed these claims, clarifying that 
        no such move is planned and urging people not to share misleading financial 
        information that may cause panic among the public."""
    )
]


    for label, text in examples:
        if st.button(label):
            st.session_state["example"] = text


# ---------------------- USER INPUT ----------------------
user_text = st.text_area(
    "Paste news article text",
    height=260,
    value=st.session_state.get("example", "")
)


# ---------------------- MODEL LOADING ----------------------
model_loaded = None
tokenizer = None
model_type = None

def lazy_load_model():
    global model_loaded, tokenizer, model_type

    if model_loaded is not None:
        return True

    # Try TensorFlow
    if USE_TF:
        try:
            tokenizer_local = DistilBertTokenizerFast.from_pretrained(model_dir)
            tf_model = TFDistilBertForSequenceClassification.from_pretrained(model_dir)
            model_loaded = tf_model
            tokenizer = tokenizer_local
            model_type = "tf"
            return True
        except Exception:
            pass

    # Try PyTorch
    if TORCH_AVAILABLE:
        try:
            tokenizer_local = AutoTokenizer.from_pretrained(model_dir)
            pt_model = AutoModelForSequenceClassification.from_pretrained(model_dir)
            model_loaded = pt_model
            tokenizer = tokenizer_local
            model_type = "torch"
            return True
        except Exception:
            pass

    return False


# ------------------------- PREDICT -------------------------
if st.button("Predict"):
    if not user_text.strip():
        st.warning("Please paste an article first.")
    else:
        if not lazy_load_model():
            st.error("Model could not be loaded. Check model directory.")
        else:
            text = clean_text(user_text)

            # forward pass
            try:
                if model_type == "tf":
                    inputs = tokenizer(text, return_tensors="tf", truncation=True, padding=True, max_length=256)
                    outputs = model_loaded(**inputs, training=False)
                    logits = outputs.logits.numpy()[0]
                else:
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    model_loaded.to(device)
                    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256).to(device)
                    with torch.no_grad():
                        out = model_loaded(**inputs)
                    logits = out.logits.cpu().numpy()[0]
            except Exception as e:
                st.error(f"Model failed to run: {e}")
                logits = None

            if logits is None:
                st.error("No prediction available.")
            else:
                exp = np.exp(logits - np.max(logits))
                probs = exp / exp.sum()

                fake_prob = float(probs[0])
                real_prob = float(probs[1])

                # UI
                st.subheader("Prediction")
                st.write("Model confidence:")

                st.progress(int(real_prob * 100))
                st.write(f"Real: **{real_prob:.4f}** | Fake: **{fake_prob:.4f}**")

                if real_prob >= threshold:
                    st.success(f"🟢 REAL NEWS — Confidence {real_prob:.4f}")
                elif fake_prob >= threshold:
                    st.error(f"🔴 FAKE NEWS — Confidence {fake_prob:.4f}")
                else:
                    st.warning(f"🟡 UNCERTAIN — Real: {real_prob:.4f}, Fake: {fake_prob:.4f}")

                # EXPLANATION
                with st.expander("Why did the model predict this? (Explanation)"):
                    token_scores = None

                    if model_type == "tf":
                        token_scores = tf_token_importances(model_loaded, tokenizer, text)

                    if token_scores is None and model_type == "torch":
                        token_scores = torch_attention_importances(model_loaded, tokenizer, text)

                    # final fallback
                    if token_scores is None:
                        tokens = tokenizer.tokenize(text)[:40]
                        heur = [(t, 1.0 / len(tokens)) for t in tokens]
                        token_scores = heur

                    reason = build_reason_from_token_scores(token_scores)
                    st.write(reason)

                    try:
                        import pandas as pd
                        df = pd.DataFrame(token_scores, columns=["token", "score"])
                        df = df.sort_values("score", ascending=False).head(12)
                        st.bar_chart(df.set_index("token"))
                    except Exception:
                        pass
