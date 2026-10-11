from pathlib import Path

import pandas as pd
from flask import Flask, request, render_template_string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ============================================================
# 1. PATHS AND SETTINGS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
CSV_FILE = BASE_DIR / "aravot_news.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.20


# ============================================================
# 2. LOAD AND CLEAN DATA
# ============================================================

if not CSV_FILE.exists():
    raise FileNotFoundError(
        f"Dataset not found: {CSV_FILE}\n"
        "Make sure aravot_news.csv is in the same folder as api.py."
    )

df = pd.read_csv(CSV_FILE)

required_columns = {"title", "text", "category"}
missing_columns = required_columns - set(df.columns)
if missing_columns:
    raise ValueError(
        "The CSV file is missing required columns: "
        + ", ".join(sorted(missing_columns))
    )

df = df.dropna(subset=["title", "text", "category"]).copy()
df["title"] = df["title"].astype(str).str.strip()
df["text"] = df["text"].astype(str).str.strip()
df["category"] = df["category"].astype(str).str.strip().str.lower()

df = df[
    (df["title"] != "")
    & (df["text"] != "")
    & (df["category"] != "")
].copy()

if "url" in df.columns:
    df = df.drop_duplicates(subset=["url"])
else:
    df = df.drop_duplicates(subset=["title", "text"])

df = df.reset_index(drop=True)

if df.empty:
    raise ValueError("No valid articles remain after cleaning the dataset.")
if df["category"].nunique() < 2:
    raise ValueError("At least two different categories are required to train the model.")

# Match the Notebook workflow: combine title + text and use TF-IDF.
X = df["title"] + " " + df["text"]
y = df["category"]

print("=" * 60)
print("ARAVOT NEWS CLASSIFIER")
print("=" * 60)
print("Dataset shape:", df.shape)
print("\nCategory distribution:")
print(y.value_counts())
print()


# ============================================================
# 3. TRAIN / TEST SPLIT
# ============================================================

# This reproduces a common 80/20 split with the same random state.
# The Notebook screenshot shows 27 test rows out of 134 total rows,
# so its exact split cannot be guaranteed from this screenshot alone.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)


# ============================================================
# 4. TF-IDF + LOGISTIC REGRESSION
# ============================================================

# Fit the vectorizer on training text only to avoid test-set leakage.
tfidf = TfidfVectorizer()
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

evaluation_model = LogisticRegression(max_iter=1000)
evaluation_model.fit(X_train_tfidf, y_train)

y_pred = evaluation_model.predict(X_test_tfidf)

print("MODEL EVALUATION (held-out test set)")
print("-" * 60)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification report:")
print(classification_report(y_test, y_pred, zero_division=0))
label_order = sorted(y.unique())
print("Confusion matrix (labels order):")
print(label_order)
print(confusion_matrix(y_test, y_pred, labels=label_order))
print()


# ============================================================
# 5. TRAIN FINAL MODEL ON ALL CLEANED DATA
# ============================================================

# Train a separate final vectorizer/model on all available data for the
# interactive web app. The held-out score above comes from the evaluation
# model and is not the score of predictions made on the training data.
tfidf = TfidfVectorizer()
X_all_tfidf = tfidf.fit_transform(X)
model = LogisticRegression(max_iter=1000)
model.fit(X_all_tfidf, y)

print("Final model trained successfully on all cleaned articles.")
print("Production model categories:", list(model.classes_))
print()


# ============================================================
# 6. FLASK APP
# ============================================================

app = Flask(__name__)

HTML = '''
<!DOCTYPE html>
<html lang="hy">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Aravot News Classifier</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background: #f2f4f7;
            margin: 0;
            padding: 40px 16px;
        }
        .container {
            max-width: 900px;
            margin: auto;
            background: white;
            padding: 35px;
            border-radius: 15px;
            box-shadow: 0 5px 20px rgba(0,0,0,0.1);
        }
        h1 { text-align: center; margin-bottom: 30px; }
        label { font-size: 18px; font-weight: bold; }
        input, textarea {
            width: 100%;
            padding: 12px;
            margin-top: 8px;
            margin-bottom: 20px;
            border: 1px solid #ccc;
            border-radius: 8px;
            font-size: 16px;
            box-sizing: border-box;
        }
        textarea { min-height: 250px; resize: vertical; }
        button {
            width: 100%;
            padding: 14px;
            background: #222;
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 18px;
            cursor: pointer;
        }
        button:hover { background: #444; }
        .result {
            margin-top: 30px;
            padding: 20px;
            background: #eef6ff;
            border-radius: 10px;
        }
        .category {
            font-size: 30px;
            font-weight: bold;
            text-align: center;
            margin: 15px;
            overflow-wrap: anywhere;
        }
        .probability { font-size: 18px; line-height: 1.8; }
        .error {
            margin-top: 20px;
            padding: 14px;
            background: #fff0f0;
            border-radius: 8px;
            color: #8b0000;
        }
        @media (max-width: 600px) {
            body { padding: 20px 10px; }
            .container { padding: 22px 16px; }
        }
    </style>
</head>
<body>
<div class="container">
    <h1>📰 Aravot News Classifier</h1>
    <form method="POST">
        <label for="title">Նորության վերնագիր</label>
        <input id="title" type="text" name="title"
               placeholder="Գրիր հոդվածի վերնագիրը..."
               value="{{ title }}" required>

        <label for="text">Նորության տեքստ</label>
        <textarea id="text" name="text"
                  placeholder="Գրիր հոդվածի տեքստը..." required>{{ news_text }}</textarea>

        <button type="submit">Դասակարգել</button>
    </form>

    {% if error %}
    <div class="error">{{ error }}</div>
    {% endif %}

    {% if prediction %}
    <div class="result">
        <h2>Մոդելի արդյունքը</h2>
        <div class="category">{{ prediction }}</div>
        <h3>Գնահատված հավանականություններ</h3>
        <div class="probability">
            {% for category, probability in probabilities.items() %}
            <div><b>{{ category }}</b>: {{ probability }}%</div>
            {% endfor %}
        </div>
    </div>
    {% endif %}
</div>
</body>
</html>
'''


# ============================================================
# 7. ROUTE
# ============================================================

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    probabilities = {}
    title = ""
    news_text = ""
    error = None

    if request.method == "POST":
        title = request.form.get("title", "").strip()
        news_text = request.form.get("text", "").strip()

        if not title or not news_text:
            error = "Խնդրում ենք լրացնել վերնագիրն ու նորության տեքստը։"
        else:
            new_article = f"{title} {news_text}"
            article_tfidf = tfidf.transform([new_article])
            prediction = model.predict(article_tfidf)[0]
            proba = model.predict_proba(article_tfidf)[0]

            probabilities = {
                category: round(float(probability) * 100, 2)
                for category, probability in zip(model.classes_, proba)
            }
            probabilities = dict(
                sorted(
                    probabilities.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )
            )

    return render_template_string(
        HTML,
        prediction=prediction,
        probabilities=probabilities,
        title=title,
        news_text=news_text,
        error=error,
    )


# ============================================================
# 8. START SERVER
# ============================================================

if __name__ == "__main__":
    # Debug mode is for local development only.
    app.run(debug=True)
