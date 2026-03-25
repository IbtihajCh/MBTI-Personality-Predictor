import gradio as gr
import joblib
import re
import numpy as np

vectorizer = joblib.load("vectorizer.pkl")
models = {
    "IE": joblib.load("IE_model.pkl"),
    "NS": joblib.load("NS_model.pkl"),
    "TF": joblib.load("TF_model.pkl"),
    "JP": joblib.load("JP_model.pkl"),
}

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def predict_mbti(text):
    if not text.strip():
        return "⚠️ Please enter some text to analyze."

    clean = clean_text(text)
    X = vectorizer.transform([clean])

    results = {}
    traits = {
        "IE": ("I", "E"),
        "NS": ("N", "S"),
        "TF": ("T", "F"),
        "JP": ("J", "P"),
    }

    for key, (pos_label, neg_label) in traits.items():
        model = models[key]
        score = model.decision_function(X)[0]
        prob = 1 / (1 + np.exp(-score))  # sigmoid
        results[key] = pos_label if prob >= 0.5 else neg_label

    mbti = results["IE"] + results["NS"] + results["TF"] + results["JP"]
    return f"🧠 **Predicted MBTI Type: {mbti}**"


mbti_info = """
### 🕊️ Analysts (Strategic Thinkers)
- **INTJ – The Architect:** Strategic, independent, loves planning ahead.  
- **INTP – The Thinker:** Curious, logical, loves deep analysis.  
- **ENTJ – The Commander:** Natural leader, confident and decisive.  
- **ENTP – The Debater:** Quick-witted innovator who loves arguments (for fun).

### 🌈 Diplomats (Empaths & Idealists)
- **INFJ – The Advocate:** Deep, visionary, values purpose and meaning.  
- **INFP – The Mediator:** Dreamy, emotional, authentic.  
- **ENFJ – The Protagonist:** Charismatic and inspiring leader.  
- **ENFP – The Campaigner:** Energetic, curious, and spontaneous.

### ⚙️ Sentinels (Organizers & Realists)
- **ISTJ – The Inspector:** Reliable, practical, and structured.  
- **ISFJ – The Defender:** Loyal, caring, quietly supportive.  
- **ESTJ – The Executive:** Organized, disciplined, loves order.  
- **ESFJ – The Caregiver:** Warm, social, community-focused.

### ⚡ Explorers (Adaptable Doers)
- **ISTP – The Virtuoso:** Hands-on problem solver, loves tinkering.  
- **ISFP – The Artist:** Creative, aesthetic, and emotional.  
- **ESTP – The Entrepreneur:** Bold, risk-taking, and practical.  
- **ESFP – The Entertainer:** Fun, energetic, and social butterfly.
"""

with gr.Blocks(theme=gr.themes.Soft(primary_hue="violet", secondary_hue="indigo")) as demo:
    gr.Markdown(
        """
        # 🧩 MBTI Personality Predictor  
        Type or paste your text below — the model will analyze your words and predict your MBTI type (like **INFJ** or **ESTP**)  
        Trained using **TF-IDF + Logistic Regression** on the MBTI Kaggle dataset.  
        """
    )

    text_input = gr.Textbox(
        label="✍️ Your Text",
        placeholder="Write a paragraph or two describing your thoughts, opinions, or experiences...",
        lines=6
    )

    predict_btn = gr.Button("🔮 Predict Personality", variant="primary")

    output_text = gr.Markdown("")

    predict_btn.click(fn=predict_mbti, inputs=text_input, outputs=output_text)

    with gr.Accordion("📘 Learn About MBTI Types", open=False):
        gr.Markdown(mbti_info)

    gr.Markdown(
        """
        ---
        <div align='center'>
        Made with ❤️ by <b>Muhammad Ibtihaj</b> | Powered by Classical ML ✨  
        Dataset: <a href="https://www.kaggle.com/datasets/datasnaek/mbti-type" target="_blank">MBTI Kaggle Dataset</a>
        </div>
        """
    )

demo.launch()
