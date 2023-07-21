import numpy as np

class ThresholdTuner:

    def __init__(y_true):
        pass

    def profit(y_true, y_pred_proba, threshold=0.5):
        y_pred = (y_pred_proba[:,1] >= threshold).astype(int)
        fp = np.sum((y_pred == 1) & (y_true == 0))
        tp = np.sum((y_pred == 1) & (y_true == 1))
        n = len(y_true)
        return (fp * -10 + tp * 100) / n
    
    def tune_threshold():
        thresholds = np.arange(0.1, 1, 0.1)
        best_threshold = 0
        best_score = -np.inf
        for t in thresholds:
            score = profit(y_true, y_pred_proba, threshold=t)
            if score > best_score:
                best_threshold = t
        return best_threshold