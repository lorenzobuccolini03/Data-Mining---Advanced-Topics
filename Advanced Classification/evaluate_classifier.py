import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    precision_recall_curve,
    average_precision_score
)
from sklearn.preprocessing import label_binarize


def evaluate_classifier(
    model,
    X_test,
    y_test,
    model_name="Model",
    labels=None,
    average="weighted",
    plot_confusion=True,
    plot_roc=True,
    plot_pr=True,
    normalize_cm=None,
    return_report=True
):
    """
    Evaluate a multi-class classifier on the test set.

    Parameters
    ----------
    model : fitted sklearn-like classifier
        Already trained classifier.
    X_test : array-like or DataFrame
        Test features.
    y_test : array-like or Series
        True test labels.
    model_name : str
        Name of the classifier.
    labels : list, optional
        Ordered class labels. If None, inferred from y_test.
    average : str
        Averaging strategy for precision/recall/F1.
        Recommended: "weighted" if classes are imbalanced, "macro" to treat all classes equally.
    plot_confusion : bool
        Whether to plot the confusion matrix.
    plot_roc : bool
        Whether to plot one-vs-rest ROC curves.
    plot_pr : bool
        Whether to plot one-vs-rest precision-recall curves.
    normalize_cm : {None, "true", "pred", "all"}
        Normalization for confusion matrix.
    return_report : bool
        Whether to return the per-class classification report.

    Returns
    -------
    results : dict
        Main aggregate metrics.
    report_df : DataFrame
        Per-class precision, recall, F1-score, support.
    """

    if labels is None:
        labels = np.unique(y_test)

    y_pred = model.predict(X_test)

    results = {
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        f"precision_{average}": precision_score(
            y_test, y_pred, labels=labels, average=average, zero_division=0
        ),
        f"recall_{average}": recall_score(
            y_test, y_pred, labels=labels, average=average, zero_division=0
        ),
        f"f1_{average}": f1_score(
            y_test, y_pred, labels=labels, average=average, zero_division=0
        ),
        "precision_macro": precision_score(
            y_test, y_pred, labels=labels, average="macro", zero_division=0
        ),
        "recall_macro": recall_score(
            y_test, y_pred, labels=labels, average="macro", zero_division=0
        ),
        "f1_macro": f1_score(
            y_test, y_pred, labels=labels, average="macro", zero_division=0
        ),
    }

    report_df = pd.DataFrame(
        classification_report(
            y_test,
            y_pred,
            labels=labels,
            output_dict=True,
            zero_division=0
        )
    ).T

    if plot_confusion:
        cm = confusion_matrix(y_test, y_pred, labels=labels, normalize=normalize_cm)

        fig, ax = plt.subplots(figsize=(7, 6))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(ax=ax, values_format=".2f" if normalize_cm else "d", cmap="Blues")
        ax.set_title(f"Confusion Matrix - {model_name}")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    has_proba = hasattr(model, "predict_proba")
    has_decision = hasattr(model, "decision_function")

    if has_proba:
        y_score = model.predict_proba(X_test)
    elif has_decision:
        y_score = model.decision_function(X_test)
    else:
        y_score = None

    if y_score is not None and len(labels) > 2:
        y_test_bin = label_binarize(y_test, classes=labels)

        try:
            results["roc_auc_ovr_weighted"] = roc_auc_score(
                y_test_bin, y_score, average="weighted", multi_class="ovr"
            )
            results["roc_auc_ovr_macro"] = roc_auc_score(
                y_test_bin, y_score, average="macro", multi_class="ovr"
            )
        except Exception:
            results["roc_auc_ovr_weighted"] = np.nan
            results["roc_auc_ovr_macro"] = np.nan

        if plot_roc:
            fig, ax = plt.subplots(figsize=(7, 6))

            for i, cls in enumerate(labels):
                RocCurveDisplay.from_predictions(
                    y_test_bin[:, i],
                    y_score[:, i],
                    name=f"{cls} vs rest",
                    ax=ax
                )

            ax.plot([0, 1], [0, 1], linestyle="--")
            ax.set_title(f"One-vs-Rest ROC Curves - {model_name}")
            plt.tight_layout()
            plt.show()

        if plot_pr:
            fig, ax = plt.subplots(figsize=(7, 6))

            for i, cls in enumerate(labels):
                PrecisionRecallDisplay.from_predictions(
                    y_test_bin[:, i],
                    y_score[:, i],
                    name=f"{cls} vs rest",
                    ax=ax
                )

            ax.set_title(f"One-vs-Rest Precision-Recall Curves - {model_name}")
            plt.tight_layout()
            plt.show()

    elif y_score is not None and len(labels) == 2:
        positive_class = labels[1]

        if y_score.ndim == 2:
            y_score_binary = y_score[:, 1]
        else:
            y_score_binary = y_score

        y_test_binary = (np.array(y_test) == positive_class).astype(int)

        results["roc_auc"] = roc_auc_score(y_test_binary, y_score_binary)
        results["average_precision"] = average_precision_score(y_test_binary, y_score_binary)

        if plot_roc:
            fig, ax = plt.subplots(figsize=(7, 6))
            RocCurveDisplay.from_predictions(y_test_binary, y_score_binary, ax=ax)
            ax.plot([0, 1], [0, 1], linestyle="--")
            ax.set_title(f"ROC Curve - {model_name}")
            plt.tight_layout()
            plt.show()

        if plot_pr:
            fig, ax = plt.subplots(figsize=(7, 6))
            PrecisionRecallDisplay.from_predictions(y_test_binary, y_score_binary, ax=ax)
            ax.set_title(f"Precision-Recall Curve - {model_name}")
            plt.tight_layout()
            plt.show()

    else:
        results["roc_auc_ovr_weighted"] = np.nan
        results["roc_auc_ovr_macro"] = np.nan
        print(f"[Warning] {model_name}: no predict_proba or decision_function available. ROC/PR curves skipped.")

    if return_report:
        print(f"\nClassification report - {model_name}")
        display(report_df)

    return results, report_df