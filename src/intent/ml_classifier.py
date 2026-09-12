import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

from src.intent.taxonomy import INTENTS
from src.intent.domain_features import get_best_domain_intent


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "data/golden/intent_review_grouped.csv"

INTENT_LABELS = list(INTENTS)


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text):
    """
    Normalize customer message text before classification.
    """

    if not isinstance(text, str):
        return ""

    text = text.lower().strip()

    # Normalize whitespace
    text = " ".join(text.split())

    return text


# ============================================================
# ML PIPELINE
# ============================================================

def create_pipeline():
    """
    Create TF-IDF + Logistic Regression pipeline.
    """

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 3),
        min_df=1,
        max_df=0.95,
        sublinear_tf=True,
        max_features=100000,
    )

    classifier = LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        C=2.0,
        solver="lbfgs",
        random_state=42,
    )

    return Pipeline([
        ("tfidf", vectorizer),
        ("classifier", classifier),
    ])


# ============================================================
# DOMAIN RULE + ML HYBRID LOGIC
# ============================================================

def apply_domain_rules(text, ml_intent, ml_confidence):
    """
    Combine ML prediction with domain-specific rules.

    Strong domain signals can override weak ML predictions.
    """

    domain_result = get_best_domain_intent(text)

    # --------------------------------------------------------
    # Handle different possible return formats
    # --------------------------------------------------------

    if isinstance(domain_result, dict):

        domain_intent = domain_result.get("intent")
        domain_score = domain_result.get("score", 0)

    elif isinstance(domain_result, tuple):

        if len(domain_result) >= 2:
            domain_intent = domain_result[0]
            domain_score = domain_result[1]
        else:
            domain_intent = None
            domain_score = 0

    else:

        domain_intent = None
        domain_score = 0

    # --------------------------------------------------------
    # Convert score safely to float
    # --------------------------------------------------------

    try:
        domain_score = float(domain_score)

    except (TypeError, ValueError):

        domain_score = 0.0

    # --------------------------------------------------------
    # No useful domain signal
    # --------------------------------------------------------

    if not domain_intent or domain_score <= 0:

        return {
            "intent": ml_intent,
            "confidence": ml_confidence,
            "source": "ML",

            "ml_intent": ml_intent,
            "ml_confidence": ml_confidence,

            "domain_intent": domain_intent,
            "domain_score": domain_score,
        }

    # --------------------------------------------------------
    # STRONG DOMAIN SIGNAL
    #
    # Score >= 2 means the domain rules have strong evidence.
    # --------------------------------------------------------

    if domain_score >= 2:

        confidence = min(
            0.95,
            0.70 + (domain_score * 0.08)
        )

        return {
            "intent": domain_intent,
            "confidence": confidence,
            "source": "DOMAIN_RULE",

            "ml_intent": ml_intent,
            "ml_confidence": ml_confidence,

            "domain_intent": domain_intent,
            "domain_score": domain_score,
        }

    # --------------------------------------------------------
    # MODERATE DOMAIN SIGNAL + ML AGREEMENT
    # --------------------------------------------------------

    if domain_score >= 1 and ml_intent == domain_intent:

        confidence = max(
            ml_confidence,
            0.55 + (domain_score * 0.05)
        )

        confidence = min(
            0.95,
            confidence
        )

        return {
            "intent": ml_intent,
            "confidence": confidence,
            "source": "ML+DOMAIN",

            "ml_intent": ml_intent,
            "ml_confidence": ml_confidence,

            "domain_intent": domain_intent,
            "domain_score": domain_score,
        }

    # --------------------------------------------------------
    # MODERATE DOMAIN SIGNAL + VERY WEAK ML
    # --------------------------------------------------------

    if domain_score >= 1 and ml_confidence < 0.45:

        confidence = min(
            0.80,
            0.55 + (domain_score * 0.05)
        )

        return {
            "intent": domain_intent,
            "confidence": confidence,
            "source": "DOMAIN_RULE",

            "ml_intent": ml_intent,
            "ml_confidence": ml_confidence,

            "domain_intent": domain_intent,
            "domain_score": domain_score,
        }

    # --------------------------------------------------------
    # Otherwise trust ML
    # --------------------------------------------------------

    return {
        "intent": ml_intent,
        "confidence": ml_confidence,
        "source": "ML",

        "ml_intent": ml_intent,
        "ml_confidence": ml_confidence,

        "domain_intent": domain_intent,
        "domain_score": domain_score,
    }


# ============================================================
# PREDICT INTENT
# ============================================================

def predict_intent(text, *args):
    """
    Predict customer intent.

    Supported formats:

        predict_intent(text, model)

    OR

        predict_intent(text, vectorizer, classifier)
    """

    text = normalize_text(text)

    # --------------------------------------------------------
    # Empty message
    # --------------------------------------------------------

    if not text:

        return {
            "intent": "GENERAL_INQUIRY",
            "confidence": 0.0,
            "source": "DEFAULT",

            "ml_intent": "GENERAL_INQUIRY",
            "ml_confidence": 0.0,

            "domain_intent": None,
            "domain_score": 0,
        }

    # ========================================================
    # CASE 1
    # Pipeline/model object
    # ========================================================

    if len(args) == 1:

        model = args[0]

        probabilities = model.predict_proba([text])[0]

        best_index = int(
            np.argmax(probabilities)
        )

        ml_intent = model.classes_[best_index]

        ml_confidence = float(
            probabilities[best_index]
        )

    # ========================================================
    # CASE 2
    # Vectorizer + classifier
    # ========================================================

    elif len(args) == 2:

        vectorizer = args[0]
        classifier = args[1]

        vector = vectorizer.transform([text])

        probabilities = classifier.predict_proba(vector)[0]

        best_index = int(
            np.argmax(probabilities)
        )

        ml_intent = classifier.classes_[best_index]

        ml_confidence = float(
            probabilities[best_index]
        )

    else:

        raise TypeError(
            "predict_intent expects either "
            "(text, model) or "
            "(text, vectorizer, classifier)"
        )

    # ========================================================
    # Apply hybrid domain logic
    # ========================================================

    return apply_domain_rules(
        text=text,
        ml_intent=ml_intent,
        ml_confidence=ml_confidence,
    )


# ============================================================
# TRAIN FINAL MODEL
# ============================================================

def train_final_model(df=None):
    """
    Train the final intent classifier.

    Can be called as:

        train_final_model()

    OR:

        train_final_model(df)

    Returns:

        vectorizer, classifier
    """

    # --------------------------------------------------------
    # Load dataset if dataframe is not provided
    # --------------------------------------------------------

    if df is None:

        df = pd.read_csv(DATA_PATH)

    else:

        df = df.copy()

    # --------------------------------------------------------
    # Remove missing values
    # --------------------------------------------------------

    df = df.dropna(
        subset=[
            "customer_message",
            "reviewed_intent",
        ]
    ).copy()

    # --------------------------------------------------------
    # Clean columns
    # --------------------------------------------------------

    df["customer_message"] = (
        df["customer_message"]
        .astype(str)
        .str.strip()
    )

    df["reviewed_intent"] = (
        df["reviewed_intent"]
        .astype(str)
        .str.strip()
    )

    # Remove empty messages

    df = df[
        df["customer_message"] != ""
    ]

    # --------------------------------------------------------
    # Normalize text
    # --------------------------------------------------------

    texts = [
        normalize_text(text)
        for text in df["customer_message"].tolist()
    ]

    labels = (
        df["reviewed_intent"]
        .tolist()
    )

    # ========================================================
    # TF-IDF
    # ========================================================

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 3),
        min_df=1,
        max_df=0.95,
        sublinear_tf=True,
        max_features=100000,
    )

    # ========================================================
    # Logistic Regression
    # ========================================================

    classifier = LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        C=2.0,
        solver="lbfgs",
        random_state=42,
    )

    # ========================================================
    # Train
    # ========================================================

    X = vectorizer.fit_transform(texts)

    classifier.fit(
        X,
        labels
    )

    return vectorizer, classifier


# ============================================================
# MODEL EVALUATION
# ============================================================

def evaluate_model():
    """
    Perform 5-fold cross-validation on the reviewed dataset.
    """

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    df = pd.read_csv(DATA_PATH)

    df = df.dropna(
        subset=[
            "customer_message",
            "reviewed_intent",
        ]
    ).copy()

    # --------------------------------------------------------
    # Clean
    # --------------------------------------------------------

    df["customer_message"] = (
        df["customer_message"]
        .astype(str)
        .str.strip()
    )

    df["reviewed_intent"] = (
        df["reviewed_intent"]
        .astype(str)
        .str.strip()
    )

    df = df[
        df["customer_message"] != ""
    ]

    # --------------------------------------------------------
    # Prepare data
    # --------------------------------------------------------

    texts = (
        df["customer_message"]
        .tolist()
    )

    labels = (
        df["reviewed_intent"]
        .tolist()
    )

    # ========================================================
    # 5-FOLD STRATIFIED CROSS VALIDATION
    # ========================================================

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    predictions = []

    actual_labels = []

    # ========================================================
    # Fold loop
    # ========================================================

    for fold_number, (
        train_index,
        test_index
    ) in enumerate(
        cv.split(texts, labels),
        start=1,
    ):

        print(
            f"Running fold {fold_number}/5..."
        )

        # ----------------------------------------------------
        # Train/test split
        # ----------------------------------------------------

        X_train = [
            texts[i]
            for i in train_index
        ]

        y_train = [
            labels[i]
            for i in train_index
        ]

        X_test = [
            texts[i]
            for i in test_index
        ]

        y_test = [
            labels[i]
            for i in test_index
        ]

        # ----------------------------------------------------
        # Create fresh model
        # ----------------------------------------------------

        model = create_pipeline()

        # ----------------------------------------------------
        # Normalize text
        # ----------------------------------------------------

        X_train = [
            normalize_text(text)
            for text in X_train
        ]

        X_test = [
            normalize_text(text)
            for text in X_test
        ]

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        model.fit(
            X_train,
            y_train
        )

        # ----------------------------------------------------
        # Predict
        # ----------------------------------------------------

        fold_predictions = (
            model.predict(X_test)
        )

        predictions.extend(
            fold_predictions
        )

        actual_labels.extend(
            y_test
        )

    # ========================================================
    # METRICS
    # ========================================================

    accuracy = accuracy_score(
        actual_labels,
        predictions
    )

    precision = precision_score(
        actual_labels,
        predictions,
        average="weighted",
        zero_division=0,
    )

    recall = recall_score(
        actual_labels,
        predictions,
        average="weighted",
        zero_division=0,
    )

    f1 = f1_score(
        actual_labels,
        predictions,
        average="weighted",
        zero_division=0,
    )

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print()
    print("=" * 70)
    print("ML INTENT CROSS-VALIDATION")
    print("=" * 70)

    print(
        f"Accuracy : {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1 Score : {f1:.4f}"
    )

    print()
    print("=" * 70)
    print("CLASSIFICATION REPORT")
    print("=" * 70)

    print(
        classification_report(
            actual_labels,
            predictions,
            labels=INTENT_LABELS,
            target_names=INTENT_LABELS,
            zero_division=0,
        )
    )

    # ========================================================
    # Return metrics
    # ========================================================

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("INTENT CLASSIFIER")
    print("=" * 70)

    results = evaluate_model()

    print()
    print("=" * 70)
    print("EVALUATION COMPLETED")
    print("=" * 70)

    print(
        f"Accuracy : {results['accuracy']:.4f}"
    )

    print(
        f"Precision: {results['precision']:.4f}"
    )

    print(
        f"Recall   : {results['recall']:.4f}"
    )

    print(
        f"F1 Score : {results['f1']:.4f}"
    )