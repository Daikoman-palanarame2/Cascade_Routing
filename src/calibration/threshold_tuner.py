import numpy as np
import pandas as pd
import json
import os
from sklearn.model_selection import TunedThresholdClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer
from src.router.classifier import DifficultyClassifier

def accuracy_floor_objective(y_true, y_pred, accuracy_floor: float = 0.90):
    """
    Custom cost-sensitive scorer:
    y_true: actual correct (1 = local correct, 0 = needs escalation)
    y_pred: prediction (1 = trust local, 0 = escalate)
    """
    # Calculate accuracy over predictions
    # If we trust the local model (y_pred=1), the correctness is y_true.
    # If we escalate (y_pred=0), we assume the remote model gets it right (accuracy=1.0).
    effective_correct = np.where(y_pred == 1, y_true, 1)
    accuracy = float(np.mean(effective_correct))
    
    # Calculate the escalation rate
    escalation_rate = float(np.mean(y_pred == 0))
    
    if accuracy < accuracy_floor:
        # Heavily penalize accuracy below floor
        return accuracy - 2.0
    # Reward lower escalation rates while maintaining the accuracy floor
    return accuracy - 0.5 * escalation_rate

def tune_threshold(dev_set_csv: str, accuracy_floor: float = 0.90, output_dir: str = "src/calibration"):
    """
    Tune the escalation threshold using TunedThresholdClassifierCV.
    """
    if not os.path.exists(dev_set_csv):
        print(f"Error: dev set file {dev_set_csv} not found.")
        return 0.72

    df = pd.read_csv(dev_set_csv)
    # Features: confidence components
    X = df[["p_easy", "agreement", "judge_score"]].values
    y = df["local_correct"].values  # 1 = local correct, 0 = needs escalation
    
    scorer = make_scorer(
        accuracy_floor_objective, 
        accuracy_floor=accuracy_floor, 
        greater_is_better=True
    )
    
    # We train a classifier on the confidence features to predict local correctness
    base_clf = LogisticRegression(max_iter=1000)
    
    # Sweep thresholds across CV splits
    # If the dataset is too small, use a smaller CV value
    cv_value = min(3, len(df))
    
    tuned = TunedThresholdClassifierCV(
        estimator=base_clf, 
        scoring=scorer, 
        cv=cv_value,
        store_cv_results=True
    )
    tuned.fit(X, y)
    
    threshold = float(tuned.best_threshold_)
    
    # Ensure directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the calibrated threshold and details
    threshold_data = {
        "threshold": threshold,
        "accuracy_floor": accuracy_floor,
        "cv_score": float(tuned.best_score_)
    }
    
    with open(os.path.join(output_dir, "threshold.json"), "w") as f:
        json.dump(threshold_data, f, indent=2)
        
    print(f"Threshold calibration completed successfully!")
    print(f"Calibrated escalation threshold: {threshold:.4f}")
    print(f"Expected CV Score (Accuracy - 0.5 * Escalation Rate): {tuned.best_score_:.4f}")
    
    return threshold

if __name__ == "__main__":
    import sys
    csv_path = "eval/dev_set.csv"
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    tune_threshold(csv_path)
