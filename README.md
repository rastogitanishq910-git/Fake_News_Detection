# Fake News Detection using Machine Learning

Paste a news article into the app and it predicts whether it's **Real** or **Fake**, along with a confidence score.

## Folder Structure

```
Fake-News-Detection/
├── app.py                # Streamlit app (paste text -> get prediction)
├── train_model.py        # Trains and saves the model
├── requirements.txt
├── README.md
├── LICENSE
├── dataset/
│   ├── Fake.csv
│   └── True.csv
├── models/                # Saved model + vectorizer (created by train_model.py)
└── src/
    ├── preprocessing.py   # Text cleaning
    ├── training.py        # TF-IDF + model training/evaluation
    ├── predict.py          # Used by the app to make predictions
    ├── utils.py             # Paths, dataset loading, saving/loading model
    └── generate_sample_data.py  # Optional: makes a small demo dataset
```

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux

pip install -r requirements.txt
```

## Usage

Train the model (a trained model is already included, so this is only needed if you want to retrain):

```bash
python train_model.py
```

Run the app:

```bash
streamlit run app.py
```

Open `http://localhost:8501`, paste an article, click **Predict**.

## Getting "No module named numpy.core" or similar errors?

This means your installed `numpy`/`scikit-learn` versions don't match the ones the model was saved with. Fix it by reinstalling exact versions:

```bash
pip install -r requirements.txt --force-reinstall
```

Or just retrain locally after installing requirements — `python train_model.py` regenerates the model using whatever versions are in your environment.

## Results

Trained on ~44,900 articles (Kaggle's Fake and Real News Dataset):

| Model | Accuracy | F1 Score |
|---|---|---|
| Random Forest (best) | 0.997 | 0.997 |
| Passive Aggressive Classifier | 0.995 | 0.995 |
| Logistic Regression | 0.989 | 0.988 |
| Multinomial Naive Bayes | 0.945 | 0.942 |

## License

MIT — see [LICENSE](LICENSE).
