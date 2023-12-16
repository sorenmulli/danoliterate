import numpy as np
from sklearn.metrics import brier_score_loss


def multiclass_brier(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    n_classes = y_prob.shape[1]
    y_true_ = np.eye(n_classes)[y_true]
    brier_scores = [brier_score_loss(y_true_[:, i], y_prob[:, i]) for i in range(n_classes)]
    return float(np.mean(brier_scores))


def ece_score(y_true: np.ndarray, y_prob: np.ndarray, n_bins=10) -> float:
    ece = 0.0
    bin_limits = np.linspace(0, 1, n_bins + 1)
    bin_width = bin_limits[1] - bin_limits[0]

    for i in range(y_prob.shape[1]):
        bin_true = y_true == i
        bin_prob = y_prob[:, i]

        bin_sums = np.zeros(n_bins)
        bin_true_sums = np.zeros(n_bins)
        bin_counts = np.zeros(n_bins)

        for j in range(len(y_true)):
            for bin_ in range(n_bins):
                if bin_limits[bin_] <= bin_prob[j] < bin_limits[bin_ + 1]:
                    bin_sums[bin_] += bin_prob[j]
                    bin_true_sums[bin_] += bin_true[j]
                    bin_counts[bin_] += 1
                    break

        for bin_ in range(n_bins):
            if bin_counts[bin_] > 0:
                avg_pred_prob = bin_sums[bin_] / bin_counts[bin_]
                true_positive_rate = bin_true_sums[bin_] / bin_counts[bin_]
                ece += (
                    np.abs(avg_pred_prob - true_positive_rate)
                    * (bin_counts[bin_] / len(y_true))
                    * bin_width
                )

    return float(ece / y_prob.shape[1])
