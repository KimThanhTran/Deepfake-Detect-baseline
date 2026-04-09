import argparse
import csv
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


METRIC_COLUMNS = ["dataset", "accuracy", "precision", "recall", "f1", "roc_auc"]
ARRAY_FILENAMES = {
    "y_true": ("y_true.npy",),
    "y_pred": ("y_pred.npy",),
    "y_prob": ("y_prob.npy",),
    "class_names": ("class_names.npy", "labels.npy", "classes.npy"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate deepfake detection predictions across multiple datasets."
    )
    parser.add_argument(
        "--datasets_root",
        type=Path,
        default=Path("datasets"),
        help="Root directory containing dataset subfolders.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs") / "prediction_evaluation",
        help="Directory to save CSV reports and figures.",
    )
    parser.add_argument(
        "--fig_width",
        type=float,
        default=9.0,
        help="Figure width for confusion matrices.",
    )
    parser.add_argument(
        "--fig_height",
        type=float,
        default=7.0,
        help="Figure height for confusion matrices.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI used when saving figures.",
    )
    return parser.parse_args()


def configure_plotting() -> None:
    sns.set_theme(context="paper", style="whitegrid", font="DejaVu Serif")
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": 12,
            "axes.titlesize": 15,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "legend.title_fontsize": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def find_existing_file(folder: Path, candidates: Sequence[str]) -> Optional[Path]:
    for name in candidates:
        candidate = folder / name
        if candidate.is_file():
            return candidate
    return None


def load_numpy_or_torch(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        array = np.load(path, allow_pickle=True)
    elif path.suffix.lower() in {".pt", ".pth"}:
        obj = torch.load(path, map_location="cpu")
        array = obj.detach().cpu().numpy() if isinstance(obj, torch.Tensor) else np.asarray(obj)
    else:
        raise ValueError(f"Unsupported file type: {path}")
    return np.asarray(array)


def load_optional_class_names(folder: Path) -> Optional[List[str]]:
    class_path = find_existing_file(folder, ARRAY_FILENAMES["class_names"])
    if class_path is None:
        return None
    values = load_numpy_or_torch(class_path)
    values = np.asarray(values).reshape(-1)
    return [str(item) for item in values.tolist()]


def squeeze_labels(array: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim == 0:
        array = array.reshape(1)
    if array.ndim > 1:
        if array.shape[-1] == 1:
            array = array.reshape(-1)
        else:
            raise ValueError(f"{name} must be 1D or have a trailing singleton dimension; got shape {array.shape}")
    return array.reshape(-1)


def infer_predictions_from_probabilities(y_prob: np.ndarray) -> np.ndarray:
    y_prob = np.asarray(y_prob)
    if y_prob.ndim == 1:
        return (y_prob >= 0.5).astype(int)
    if y_prob.ndim == 2 and y_prob.shape[1] == 1:
        return (y_prob.reshape(-1) >= 0.5).astype(int)
    if y_prob.ndim == 2:
        return np.argmax(y_prob, axis=1)
    raise ValueError(f"Unsupported y_prob shape: {y_prob.shape}")


def prepare_probability_array(y_prob: np.ndarray, num_samples: int) -> np.ndarray:
    y_prob = np.asarray(y_prob)
    if y_prob.ndim == 0:
        raise ValueError("y_prob cannot be scalar.")
    if y_prob.shape[0] != num_samples:
        raise ValueError(
            f"y_prob first dimension ({y_prob.shape[0]}) does not match number of samples ({num_samples})."
        )
    if y_prob.ndim == 1:
        return y_prob.reshape(-1)
    if y_prob.ndim == 2 and y_prob.shape[1] == 1:
        return y_prob.reshape(-1)
    if y_prob.ndim == 2:
        row_sums = y_prob.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-3):
            exp_shifted = np.exp(y_prob - np.max(y_prob, axis=1, keepdims=True))
            softmax_prob = exp_shifted / np.clip(exp_shifted.sum(axis=1, keepdims=True), 1e-12, None)
            return softmax_prob
        return y_prob
    raise ValueError(f"Unsupported y_prob shape: {y_prob.shape}")


def sanitize_class_names(class_names: Optional[List[str]], labels: np.ndarray) -> List[str]:
    unique_labels = sorted(np.unique(labels).tolist())
    if class_names is not None and len(class_names) >= len(unique_labels):
        return [str(class_names[label]) if isinstance(label, (int, np.integer)) and label < len(class_names) else str(label) for label in unique_labels]
    return [str(label) for label in unique_labels]


def compute_roc_auc(
    y_true: np.ndarray,
    y_prob: Optional[np.ndarray],
    unique_labels: np.ndarray,
) -> Optional[float]:
    if y_prob is None:
        return None

    try:
        if len(unique_labels) == 2:
            if y_prob.ndim == 1:
                return float(roc_auc_score(y_true, y_prob))
            if y_prob.ndim == 2 and y_prob.shape[1] == 2:
                return float(roc_auc_score(y_true, y_prob[:, 1]))
            if y_prob.ndim == 2 and y_prob.shape[1] == 1:
                return float(roc_auc_score(y_true, y_prob.reshape(-1)))
        if y_prob.ndim == 2 and y_prob.shape[1] == len(unique_labels):
            return float(
                roc_auc_score(
                    y_true,
                    y_prob,
                    multi_class="ovr",
                    average="macro",
                    labels=unique_labels,
                )
            )
    except ValueError as exc:
        warnings.warn(f"Skipping ROC-AUC: {exc}")
        return None

    warnings.warn(
        "Skipping ROC-AUC because y_prob shape is incompatible with the inferred task."
    )
    return None


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Sequence[str],
    dataset_name: str,
    output_path: Path,
    normalized: bool,
    fig_size: Tuple[float, float],
    dpi: int,
) -> None:
    plt.figure(figsize=fig_size)
    fmt = ".2f" if normalized else "d"
    cmap = sns.color_palette("Blues", as_cmap=True)
    heatmap = sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        cbar=True,
        square=True,
        linewidths=0.8,
        linecolor="white",
        annot_kws={"fontsize": 11, "fontweight": "semibold"},
    )
    heatmap.set_xticklabels(class_names, rotation=20, ha="right")
    heatmap.set_yticklabels(class_names, rotation=0)
    plt.xlabel("Predicted Label", fontweight="semibold")
    plt.ylabel("True Label", fontweight="semibold")
    suffix = "Normalized Confusion Matrix" if normalized else "Confusion Matrix"
    plt.title(f"{dataset_name} - {suffix}", fontweight="bold", pad=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()


def plot_metric_comparison(results: List[Dict[str, Optional[float]]], output_path: Path, dpi: int) -> None:
    valid_metrics = [column for column in METRIC_COLUMNS[1:] if any(row.get(column) is not None for row in results)]
    if not results or not valid_metrics:
        return

    dataset_names = [row["dataset"] for row in results]
    x = np.arange(len(dataset_names))
    width = min(0.8 / max(len(valid_metrics), 1), 0.18)
    palette = sns.color_palette("deep", n_colors=len(valid_metrics))

    plt.figure(figsize=(max(12, 1.4 * len(dataset_names)), 7.5))
    for idx, metric in enumerate(valid_metrics):
        values = [row.get(metric) if row.get(metric) is not None else np.nan for row in results]
        offset = (idx - (len(valid_metrics) - 1) / 2.0) * width
        bars = plt.bar(
            x + offset,
            values,
            width=width,
            label=metric.upper(),
            color=palette[idx],
            edgecolor="black",
            linewidth=0.6,
            alpha=0.92,
        )
        for bar, value in zip(bars, values):
            if np.isnan(value):
                continue
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                value + 0.01,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
                rotation=90,
            )

    plt.xticks(x, dataset_names, rotation=30, ha="right")
    plt.ylabel("Score", fontweight="semibold")
    plt.xlabel("Dataset", fontweight="semibold")
    plt.ylim(0.0, 1.08)
    plt.title("Cross-Dataset Metric Comparison", fontweight="bold", pad=14)
    plt.legend(title="Metrics", frameon=True)
    plt.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()


def write_results_csv(results: List[Dict[str, Optional[float]]], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_COLUMNS)
        writer.writeheader()
        for row in results:
            serializable = {
                key: ("" if row.get(key) is None else row.get(key))
                for key in METRIC_COLUMNS
            }
            writer.writerow(serializable)


def save_classification_report(report_text: str, output_path: Path) -> None:
    output_path.write_text(report_text + "\n", encoding="utf-8")


def evaluate_dataset(
    dataset_dir: Path,
    output_dir: Path,
    fig_size: Tuple[float, float],
    dpi: int,
) -> Optional[Dict[str, Optional[float]]]:
    dataset_name = dataset_dir.name
    y_true_path = find_existing_file(dataset_dir, ARRAY_FILENAMES["y_true"])
    y_pred_path = find_existing_file(dataset_dir, ARRAY_FILENAMES["y_pred"])
    y_prob_path = find_existing_file(dataset_dir, ARRAY_FILENAMES["y_prob"])

    if y_true_path is None:
        print(f"[WARN] Skipping {dataset_name}: missing y_true.npy")
        return None
    if y_pred_path is None and y_prob_path is None:
        print(f"[WARN] Skipping {dataset_name}: missing both y_pred.npy and y_prob.npy")
        return None

    dataset_output_dir = output_dir / dataset_name
    dataset_output_dir.mkdir(parents=True, exist_ok=True)

    y_true = squeeze_labels(load_numpy_or_torch(y_true_path), "y_true")
    y_prob = None
    if y_prob_path is not None:
        y_prob = prepare_probability_array(load_numpy_or_torch(y_prob_path), len(y_true))

    if y_pred_path is not None:
        y_pred = squeeze_labels(load_numpy_or_torch(y_pred_path), "y_pred")
    elif y_prob is not None:
        y_pred = infer_predictions_from_probabilities(y_prob)
        print(f"[INFO] {dataset_name}: inferred y_pred from y_prob.")
    else:
        print(f"[WARN] Skipping {dataset_name}: unable to determine y_pred.")
        return None

    if len(y_true) != len(y_pred):
        print(
            f"[WARN] Skipping {dataset_name}: y_true length ({len(y_true)}) "
            f"does not match y_pred length ({len(y_pred)})."
        )
        return None

    unique_labels = np.unique(np.concatenate([y_true, y_pred]))
    class_names = sanitize_class_names(load_optional_class_names(dataset_dir), unique_labels)

    accuracy = float(accuracy_score(y_true, y_pred))
    precision = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    recall = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    roc_auc = compute_roc_auc(y_true, y_prob, unique_labels)

    cm_raw = confusion_matrix(y_true, y_pred, labels=unique_labels)
    row_sums = cm_raw.sum(axis=1, keepdims=True)
    cm_norm = np.divide(
        cm_raw.astype(np.float64),
        row_sums,
        out=np.zeros_like(cm_raw, dtype=np.float64),
        where=row_sums != 0,
    )

    report_text = classification_report(
        y_true,
        y_pred,
        labels=unique_labels,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )

    plot_confusion_matrix(
        cm=cm_raw,
        class_names=class_names,
        dataset_name=dataset_name,
        output_path=dataset_output_dir / f"{dataset_name}_confusion_matrix_raw.png",
        normalized=False,
        fig_size=fig_size,
        dpi=dpi,
    )
    plot_confusion_matrix(
        cm=cm_norm,
        class_names=class_names,
        dataset_name=dataset_name,
        output_path=dataset_output_dir / f"{dataset_name}_confusion_matrix_normalized.png",
        normalized=True,
        fig_size=fig_size,
        dpi=dpi,
    )
    save_classification_report(report_text, dataset_output_dir / f"{dataset_name}_classification_report.txt")

    print(f"\n{'=' * 80}")
    print(f"Dataset: {dataset_name}")
    print(f"Samples: {len(y_true)}")
    print(f"Classes: {len(unique_labels)}")
    print(f"Accuracy           : {accuracy:.4f}")
    print(f"Precision (macro)  : {precision:.4f}")
    print(f"Recall (macro)     : {recall:.4f}")
    print(f"F1-score (macro)   : {f1:.4f}")
    if roc_auc is None:
        print("ROC-AUC            : N/A")
    else:
        print(f"ROC-AUC            : {roc_auc:.4f}")
    print("\nClassification Report")
    print(report_text)

    return {
        "dataset": dataset_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
    }


def list_dataset_directories(root: Path) -> List[Path]:
    if not root.exists():
        return []
    return [path for path in sorted(root.iterdir()) if path.is_dir()]


def main() -> None:
    args = parse_args()
    configure_plotting()

    dataset_dirs = list_dataset_directories(args.datasets_root)
    if not dataset_dirs:
        print(f"[WARN] No dataset directories found under: {args.datasets_root.resolve()}")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results: List[Dict[str, Optional[float]]] = []

    for dataset_dir in dataset_dirs:
        result = evaluate_dataset(
            dataset_dir=dataset_dir,
            output_dir=args.output_dir,
            fig_size=(args.fig_width, args.fig_height),
            dpi=args.dpi,
        )
        if result is not None:
            results.append(result)

    if not results:
        print("[WARN] No datasets were evaluated successfully.")
        return

    results_csv = args.output_dir / "evaluation_results.csv"
    write_results_csv(results, results_csv)
    plot_metric_comparison(results, args.output_dir / "dataset_metric_comparison.png", args.dpi)

    print(f"\nSaved summary CSV to: {results_csv.resolve()}")
    print(f"Saved comparison figure to: {(args.output_dir / 'dataset_metric_comparison.png').resolve()}")


if __name__ == "__main__":
    main()
