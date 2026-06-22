# 🧩 MBTI Personality Predictor

A machine learning-powered web application that predicts your **MBTI (Myers-Briggs Type Indicator)** personality type based on text input. Built with classical ML (TF-IDF + Logistic Regression) and deployed via an interactive Gradio interface.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Dataset](#dataset)
- [Model Training](#model-training)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Results](#results)
- [Screenshots](#screenshots)
- [Technologies Used](#technologies-used)
- [Future Improvements](#future-improvements)
- [Team](#team)
- [License](#license)

---

## 🎯 Overview

The MBTI Personality Predictor analyzes user-written text to classify personality across four binary dimensions:

| Dimension | Trait 1 | Trait 2 | Description |
|-----------|---------|---------|-------------|
| **I/E** | Introversion (I) | Extraversion (E) | Energy source: internal vs. external |
| **N/S** | Intuition (N) | Sensing (S) | Information gathering: patterns vs. facts |
| **T/F** | Thinking (T) | Feeling (F) | Decision making: logic vs. values |
| **J/P** | Judging (J) | Perceiving (P) | Lifestyle: structure vs. flexibility |

**Example:** `INTJ` = Introverted, Intuitive, Thinking, Judging

---

## ✨ Features

- 🧠 **4 Independent Binary Classifiers** — One model per MBTI dimension
- 📝 **TF-IDF Vectorization** — Bigram text features (1-2 word combinations)
- ⚖️ **Class Balancing** — Handles imbalanced personality distributions
- 🎨 **Beautiful Gradio UI** — Soft violet/indigo theme with accordion info panel
- 📊 **Probability Scoring** — Uses decision function + sigmoid for confidence
- 📚 **MBTI Encyclopedia** — Built-in reference for all 16 personality types
- 💾 **Model Persistence** — Save/load trained models with `joblib`
- 🚀 **One-Click Launch** — Instant web interface with `demo.launch()`

---

## 🏗️ Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   User Input    │────▶│  Text Cleaning   │────▶│  TF-IDF Vector  │
│   (Raw Text)    │     │  (Lower, Regex)  │     │  (5000 features)│
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                                          │
                              ┌───────────────────────────┼───────────┐
                              ▼                           ▼           ▼
                        ┌─────────┐                ┌─────────┐ ┌─────────┐
                        │ IE Model│                │ NS Model│ │ TF Model│
                        │(LogReg) │                │(LogReg) │ │(LogReg) │
                        └────┬────┘                └────┬────┘ └────┬────┘
                             │                          │           │
                             └──────────────────────────┼───────────┘
                                                        ▼
                                                   ┌─────────┐
                                                   │ JP Model│
                                                   │(LogReg) │
                                                   └────┬────┘
                                                        │
                                                        ▼
                                              ┌─────────────────┐
                                              │  MBTI Assembly  │
                                              │  I + N + T + J  │
                                              │   = INTJ        │
                                              └─────────────────┘
```

---

## 🚀 Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/mbti-personality-predictor.git
cd mbti-personality-predictor

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### requirements.txt
```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
joblib>=1.1.0
tqdm>=4.62.0
gradio>=3.0.0
```

---

## 📊 Dataset

- **Source:** [Kaggle MBTI Dataset](https://www.kaggle.com/datasets/datasnaek/mbti-type)
- **Format:** CSV with columns `type` (MBTI label) and `posts` (user text)
- **Size:** ~8,600 rows of labeled personality data
- **Preprocessing:**
  - Lowercase conversion
  - URL removal (`http://...`)
  - Pipe character normalization (`||` → space)
  - Special character stripping (keep only a-z and spaces)
  - Whitespace normalization

---

## 🏋️ Model Training

### Training Pipeline (`main.ipynb`)

```python
# 1. Load & Clean Data
df = pd.read_csv("mbti_1.csv")
df['clean_posts'] = df['posts'].apply(clean_text)

# 2. Encode Labels
df['IE'] = df['type'].apply(lambda x: 1 if x[0]=='I' else 0)
df['NS'] = df['type'].apply(lambda x: 1 if x[1]=='N' else 0)
df['TF'] = df['type'].apply(lambda x: 1 if x[2]=='T' else 0)
df['JP'] = df['type'].apply(lambda x: 1 if x[3]=='J' else 0)

# 3. Vectorize Text
vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words='english',
    ngram_range=(1,2)  # Unigrams + Bigrams
)
X = vectorizer.fit_transform(df['clean_posts'])

# 4. Train 4 Models (One per trait)
for trait in ['IE', 'NS', 'TF', 'JP']:
    model = LogisticRegression(
        max_iter=300,
        class_weight='balanced',
        solver='lbfgs'
    )
    model.fit(X_train, y_train)
    joblib.dump(model, f"{trait}_model.pkl")
```

### Hyperparameters

| Parameter | Value | Reason |
|-----------|-------|--------|
| `max_features` | 5000 | Balance between coverage and speed |
| `ngram_range` | (1,2) | Capture word pairs for context |
| `max_iter` | 300 | Ensure convergence |
| `class_weight` | 'balanced' | Handle imbalanced classes |
| `solver` | 'lbfgs' | Efficient for small datasets |
| `test_size` | 0.2 | 80/20 train-test split |
| `random_state` | 42 | Reproducibility |

---

## 🎮 Usage

### Training the Models
```bash
python main.ipynb
```
This will:
- Load and preprocess the dataset
- Train 4 logistic regression models
- Save models as `.pkl` files
- Display accuracy and classification reports

### Launching the Web App
```bash
python app.py
```
This will:
- Load pre-trained models and vectorizer
- Start Gradio interface at `http://localhost:7860`
- Open browser automatically

### Using the Interface
1. **Enter Text** — Write a paragraph about your thoughts, opinions, or experiences
2. **Click Predict** — The model analyzes your text
3. **View Result** — See your predicted MBTI type (e.g., **INFJ**, **ESTP**)
4. **Explore Types** — Expand the accordion to learn about all 16 personalities

---

## 📁 Project Structure

```
mbti-personality-predictor/
│
├── main.ipynb              # Model training script
├── app.py                # Gradio web application
│
├── mbti_1.csv            # Dataset (not included, download from Kaggle)
├──models
    ├── vectorizer.pkl        # Saved TF-IDF vectorizer (auto-generated)
    ├── IE_model.pkl          # Introversion/Extraversion model
    ├── NS_model.pkl          # Intuition/Sensing model
    ├── TF_model.pkl          # Thinking/Feeling model
    └── JP_model.pkl          # Judging/Perceiving model
│
├── README.md             # Project documentation

```

---

## 📈 Results

### Model Performance

| Trait | Classes | Typical Accuracy | Notes |
|-------|---------|-----------------|-------|
| **I/E** | I vs E | ~75-80% | Most distinguishable trait |
| **N/S** | N vs S | ~85-90% | Strong linguistic signals |
| **T/F** | T vs F | ~70-75% | Moderate separability |
| **J/P** | J vs P | ~65-70% | Hardest to predict |

> *Note: Exact scores depend on random seed and data split. Run `main.ipynb` to see your specific results.*

### Classification Report Example (I/E)
```
              precision    recall  f1-score   support

           0       0.78      0.82      0.80      1200
           1       0.81      0.77      0.79      1200

    accuracy                           0.79      2400
   macro avg       0.79      0.79      0.79      2400
weighted avg       0.79      0.79      0.79      2400
```

---

## 🖼️ Screenshots

<img width="1513" height="831" alt="image" src="https://github.com/user-attachments/assets/d043194b-bd71-461a-be9e-53b712fb7a83" />


---

## 🛠️ Technologies Used

| Technology | Purpose |
|------------|---------|
| **Python 3.8+** | Core language |
| **Pandas** | Data manipulation and CSV handling |
| **NumPy** | Numerical operations and sigmoid calculation |
| **Scikit-learn** | TF-IDF vectorization, Logistic Regression, metrics |
| **Joblib** | Model serialization and deserialization |
| **Tqdm** | Progress bars for training loops |
| **Gradio** | Interactive web UI framework |
| **Regex** | Text cleaning and preprocessing |

---

## 🔮 Future Improvements

- [ ] **Deep Learning Models** — Try BERT/RoBERTa for better text understanding
- [ ] **Confidence Scores** — Show probability percentages for each dimension
- [ ] **Multi-label Classification** — Train a single model for all 16 types
- [ ] **Larger Dataset** — Incorporate more diverse text sources (Reddit, Twitter)
- [ ] **Real-time Analysis** — Live personality tracking as user types
- [ ] **Visualization** — Radar charts showing personality dimension scores
- [ ] **API Deployment** — FastAPI backend for mobile app integration
- [ ] **Docker Containerization** — Easy deployment anywhere
- [ ] **A/B Testing** — Compare classical ML vs. transformer performance

---

## 👥 Team

**Developed by:**

| Name | Role |
|------|------|
| **Muhammad Ibtihaj** 🧑🏻 | ML Engineer & Full-Stack Developer |

---

## 📝 Notes

- **Model Files:** The `.pkl` files are binary and not version-controlled. Run `main.ipynb` to generate them.
- **Dataset:** Download `mbti_1.csv` from [Kaggle](https://www.kaggle.com/datasets/datasnaek/mbti-type) before training.
- **Memory:** TF-IDF with 5000 features requires ~500MB RAM during training.
- **Inference:** Prediction is near-instant (<100ms) on modern hardware.
- **Privacy:** All text processing happens locally. No data is sent to external servers.

---

## 📄 License

This project is open-source and available for educational and personal use. Please cite the original Kaggle dataset when using this work.

---

## 🙏 Acknowledgments

- [Kaggle MBTI Dataset](https://www.kaggle.com/datasets/datasnaek/mbti-type) by DataSnaek
- Scikit-learn documentation and community
- Gradio team for the amazing UI framework
- Myers & Briggs Foundation for the MBTI framework

---

<p align="center">
  <b>Made with ❤️ and Classical ML ✨</b><br>
  <i>Discover your personality, one word at a time! 🧠</i>
</p>
