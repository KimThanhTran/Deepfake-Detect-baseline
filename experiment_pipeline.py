import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from networks import build_detector


MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SUMMARY_COLUMNS = ["model", "dataset", "accuracy", "precision", "recall", "f1", "roc_auc"]
BENCHMARK_INFERENCE_CONFIG = {
    "ForenSynths": {"no_resize": False, "no_crop": True},
    "GANGen": {"no_resize": True, "no_crop": True},
    "GANGen-Detection": {"no_resize": True, "no_crop": True},
    "DiffusionForensics": {"no_resize": False, "no_crop": True},
    "UniversalFakeDetect": {"no_resize": False, "no_crop": True},
    "GAN8": {"no_resize": False, "no_crop": True},
    "Diffusion1kStep": {"no_resize": False, "no_crop": True},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Full experiment pipeline for baseline, NPR, and NPR fine-tuning."
    )
    parser.add_argument("--datasets_root", type=Path, default=None)
    parser.add_argument("--checkpoints_dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--outputs_dir", type=Path, default=Path("outputs"))
    parser.add_argument("--models", nargs="+", default=["baseline", "npr", "npr_finetune"])
    parser.add_argument("--dataset_names", nargs="*", default=None)
    parser.add_argument("--train_dataset", default="ForenSynths")
    parser.add_argument("--train_split", default="train")
    parser.add_argument("--val_split", default="val")
    parser.add_argument("--train_name_prefix", default="paper")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--load_size", type=int, default=256)
    parser.add_argument("--crop_size", type=int, default=224)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--gpu_ids", default="0")
    parser.add_argument("--train_lr", type=float, default=1e-4)
    parser.add_argument("--finetune_lr", type=float, default=1e-5)
    parser.add_argument("--train_epochs", type=int, default=100)
    parser.add_argument("--finetune_extra_epochs", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--stages",
        nargs="+",
        default=["train", "finetune", "inference", "evaluate", "compare"],
        choices=["train", "finetune", "inference", "evaluate", "compare"],
    )
    parser.add_argument(
        "--existing_checkpoint",
        action="append",
        default=[],
        help="Optional mapping model=checkpoint_path. Example: npr=checkpoints/run/model_epoch_best.pth",
    )
    return parser.parse_args()


def configure_plotting() -> None:
    sns.set_theme(context="paper", style="whitegrid", font="DejaVu Serif")
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "legend.title_fontsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def resolve_datasets_root(explicit_root: Optional[Path]) -> Path:
    if explicit_root is not None:
        return explicit_root
    for candidate in [Path("datasets"), Path("dataset")]:
        if candidate.exists():
            return candidate
    return Path("datasets")


def parse_existing_checkpoint_args(pairs: Sequence[str]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for item in pairs:
        if "=" not in item:
            raise ValueError(f"Invalid --existing_checkpoint value: {item}")
        key, value = item.split("=", 1)
        mapping[key.strip()] = value.strip()
    return mapping


def list_dirs(path: Path) -> List[Path]:
    if not path.exists():
        return []
    return [child for child in sorted(path.iterdir()) if child.is_dir()]


def contains_binary_labels(path: Path) -> bool:
    return (path / "0_real").is_dir() and (path / "1_fake").is_dir()


def default_inference_config(dataset_root: Path) -> Dict[str, bool]:
    return BENCHMARK_INFERENCE_CONFIG.get(
        dataset_root.name,
        {"no_resize": False, "no_crop": False},
    )


def collect_dataset_targets(
    current_root: Path,
    benchmark_name: str,
    benchmark_root: Optional[Path] = None,
) -> List[Tuple[str, Path, Dict[str, bool]]]:
    benchmark_root = benchmark_root or current_root
    config = default_inference_config(benchmark_root)
    targets: List[Tuple[str, Path, Dict[str, bool]]] = []

    if contains_binary_labels(current_root):
        rel_name = current_root.relative_to(benchmark_root).as_posix().replace("/", "__")
        dataset_name = benchmark_name if rel_name == "." else f"{benchmark_name}__{rel_name}"
        return [(dataset_name, current_root, config)]

    for child in list_dirs(current_root):
        if contains_binary_labels(child):
            rel_name = child.relative_to(benchmark_root).as_posix().replace("/", "__")
            targets.append((f"{benchmark_name}__{rel_name}", child, config))
        else:
            targets.extend(collect_dataset_targets(child, benchmark_name, benchmark_root))
    return targets


def discover_dataset_targets(
    datasets_root: Path,
    dataset_names: Optional[Sequence[str]] = None,
) -> List[Tuple[str, Path, Dict[str, bool]]]:
    if not datasets_root.exists():
        raise FileNotFoundError(f"Datasets root not found: {datasets_root}")

    selected = set(dataset_names) if dataset_names else None
    targets: List[Tuple[str, Path, Dict[str, bool]]] = []
    for benchmark_root in list_dirs(datasets_root):
        if selected and benchmark_root.name not in selected:
            continue
        targets.extend(collect_dataset_targets(benchmark_root, benchmark_root.name))
    return targets


def get_training_root(datasets_root: Path, dataset_name: str, split: str) -> Path:
    root = datasets_root / dataset_name
    if not root.exists():
        raise FileNotFoundError(f"Training dataset root not found: {root}")

    split_root = root / split
    if split_root.exists():
        return root
    if contains_binary_labels(root):
        return datasets_root
    raise FileNotFoundError(
        f"Could not find training split '{split}' under {root} and root is not directly binary-labeled."
    )


def build_train_command(
    model_type: str,
    run_name: str,
    dataroot: Path,
    args: argparse.Namespace,
    continue_train: bool = False,
    lr: Optional[float] = None,
    epoch: str = "latest",
    niter: Optional[int] = None,
) -> List[str]:
    command = [
        sys.executable,
        "train.py",
        "--model_type",
        model_type,
        "--name",
        run_name,
        "--dataroot",
        str(dataroot),
        "--train_split",
        args.train_split,
        "--val_split",
        args.val_split,
        "--batch_size",
        str(args.batch_size),
        "--loadSize",
        str(args.load_size),
        "--cropSize",
        str(args.crop_size),
        "--gpu_ids",
        args.gpu_ids,
        "--num_threads",
        str(args.num_workers),
        "--checkpoints_dir",
        str(args.checkpoints_dir),
        "--lr",
        str(lr if lr is not None else args.train_lr),
        "--niter",
        str(niter if niter is not None else args.train_epochs),
    ]
    if continue_train:
        command.extend(["--continue_train", "--epoch", epoch])
    return command


def latest_checkpoint_in_run(run_dir: Path) -> Optional[Path]:
    candidate = run_dir / "model_epoch_latest.pth"
    if candidate.is_file():
        return candidate
    best_candidate = run_dir / "model_epoch_best.pth"
    if best_candidate.is_file():
        return best_candidate
    return None


def snapshot_checkpoint(source_checkpoint: Path, outputs_dir: Path, model_key: str) -> Path:
    snapshot_dir = outputs_dir / "model_snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    target_checkpoint = snapshot_dir / f"{model_key}.pth"
    shutil.copy2(source_checkpoint, target_checkpoint)
    return target_checkpoint


def detect_new_run_dir(checkpoints_dir: Path, prefix: str, before: Sequence[Path]) -> Path:
    before_names = {item.name for item in before}
    candidates = [
        path for path in list_dirs(checkpoints_dir)
        if path.name.startswith(prefix) and path.name not in before_names
    ]
    if not candidates:
        existing = [path for path in list_dirs(checkpoints_dir) if path.name.startswith(prefix)]
        if existing:
            return max(existing, key=lambda path: path.stat().st_mtime)
        raise FileNotFoundError(f"Could not find a checkpoint directory starting with '{prefix}'.")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def run_subprocess(command: Sequence[str]) -> None:
    print("Running:", " ".join(command))
    subprocess.run(command, check=True)


def load_checkpoint_metadata(checkpoint_path: Path) -> Dict[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict):
        return checkpoint
    return {"model": checkpoint}


def train_model(
    model_key: str,
    model_type: str,
    args: argparse.Namespace,
    datasets_root: Path,
    state: Dict[str, Dict[str, str]],
) -> None:
    if model_key in state and Path(state[model_key]["checkpoint"]).is_file():
        print(f"[INFO] Reusing existing checkpoint for {model_key}: {state[model_key]['checkpoint']}")
        return

    dataroot = get_training_root(datasets_root, args.train_dataset, args.train_split)
    run_prefix = f"{args.train_name_prefix}_{model_key}"
    before = list_dirs(args.checkpoints_dir)
    command = build_train_command(
        model_type=model_type,
        run_name=run_prefix,
        dataroot=dataroot,
        args=args,
    )
    run_subprocess(command)
    run_dir = detect_new_run_dir(args.checkpoints_dir, run_prefix, before)
    checkpoint = latest_checkpoint_in_run(run_dir)
    if checkpoint is None:
        raise FileNotFoundError(f"No checkpoint found after training in {run_dir}")
    snapshot = snapshot_checkpoint(checkpoint, args.outputs_dir, model_key)

    state[model_key] = {
        "model_type": model_type,
        "run_dir": str(run_dir),
        "checkpoint": str(snapshot),
    }


def finetune_model(
    args: argparse.Namespace,
    state: Dict[str, Dict[str, str]],
) -> None:
    if "npr_finetune" in state and Path(state["npr_finetune"]["checkpoint"]).is_file():
        print(f"[INFO] Reusing existing checkpoint for npr_finetune: {state['npr_finetune']['checkpoint']}")
        return

    npr_state = state.get("npr")
    if not npr_state:
        raise FileNotFoundError("NPR checkpoint metadata is missing; train or provide NPR first.")

    run_dir = Path(npr_state["run_dir"])
    latest_path = run_dir / "model_epoch_latest.pth"
    if not latest_path.is_file():
        raise FileNotFoundError(f"NPR latest checkpoint not found: {latest_path}")

    metadata = load_checkpoint_metadata(latest_path)
    start_epoch = int(metadata.get("epoch", 0))
    target_niter = start_epoch + args.finetune_extra_epochs
    command = build_train_command(
        model_type="npr",
        run_name=run_dir.name,
        dataroot=Path(get_training_root(resolve_datasets_root(args.datasets_root), args.train_dataset, args.train_split)),
        args=args,
        continue_train=True,
        lr=args.finetune_lr,
        epoch="latest",
        niter=target_niter,
    )
    run_subprocess(command)

    checkpoint = latest_checkpoint_in_run(run_dir)
    if checkpoint is None:
        raise FileNotFoundError(f"No fine-tuned checkpoint found in {run_dir}")
    snapshot = snapshot_checkpoint(checkpoint, args.outputs_dir, "npr_finetune")

    state["npr_finetune"] = {
        "model_type": "npr",
        "run_dir": str(run_dir),
        "checkpoint": str(snapshot),
    }


def normalize_state_with_existing_checkpoints(
    args: argparse.Namespace,
    state: Dict[str, Dict[str, str]],
) -> Dict[str, Dict[str, str]]:
    mapping = parse_existing_checkpoint_args(args.existing_checkpoint)
    for model_key, checkpoint_str in mapping.items():
        checkpoint_path = Path(checkpoint_str)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found for {model_key}: {checkpoint_path}")
        metadata = load_checkpoint_metadata(checkpoint_path)
        model_type = str(metadata.get("model_type", "npr"))
        state[model_key] = {
            "model_type": model_type,
            "run_dir": str(checkpoint_path.parent),
            "checkpoint": str(checkpoint_path),
        }
    return state


def load_model(model_path: Path, model_type: str, device: torch.device) -> torch.nn.Module:
    checkpoint = torch.load(model_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_type" in checkpoint:
        model_type = str(checkpoint["model_type"])
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    first_key = next(iter(state_dict))
    if first_key.startswith("module."):
        state_dict = {key.replace("module.", "", 1): value for key, value in state_dict.items()}

    model = build_detector(model_type=model_type, num_classes=1)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def list_images(folder: Path) -> List[Path]:
    return [
        path for path in sorted(folder.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def collect_binary_samples(root: Path) -> List[Tuple[Path, int]]:
    real_dir = root / "0_real"
    fake_dir = root / "1_fake"
    if not real_dir.is_dir() or not fake_dir.is_dir():
        raise ValueError(f"Expected 0_real and 1_fake under {root}")

    samples: List[Tuple[Path, int]] = []
    for image_path in list_images(real_dir):
        samples.append((image_path, 0))
    for image_path in list_images(fake_dir):
        samples.append((image_path, 1))
    if not samples:
        raise ValueError(f"No images found in {root}")
    return samples


def build_transform(load_size: int, crop_size: int, no_resize: bool, no_crop: bool) -> transforms.Compose:
    resize_transform = transforms.Lambda(lambda img: img) if no_resize else transforms.Resize((load_size, load_size))
    crop_transform = transforms.Lambda(lambda img: img) if no_crop else transforms.CenterCrop(crop_size)
    return transforms.Compose(
        [
            resize_transform,
            crop_transform,
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
        ]
    )


class BinaryImageDataset(Dataset):
    def __init__(self, root: Path, transform: transforms.Compose):
        self.root = root
        self.samples = collect_binary_samples(root)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        try:
            image = Image.open(path).convert("RGB")
            tensor = self.transform(image)
            return tensor, label, str(path)
        except Exception as exc:
            print(f"[WARN] Skipping unreadable image {path}: {exc}")
            return None


def safe_collate(batch):
    items = [item for item in batch if item is not None]
    if not items:
        return None
    images = torch.stack([item[0] for item in items], dim=0)
    labels = torch.tensor([item[1] for item in items], dtype=torch.long)
    paths = [item[2] for item in items]
    return images, labels, paths


def save_prediction_csv(output_dir: Path, paths: Sequence[str], y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> None:
    csv_path = output_dir / "predictions.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["path", "label", "score_fake", "prediction", "correct"],
        )
        writer.writeheader()
        for path, label, pred, prob in zip(paths, y_true, y_pred, y_prob):
            writer.writerow(
                {
                    "path": path,
                    "label": int(label),
                    "score_fake": float(prob),
                    "prediction": int(pred),
                    "correct": int(label == pred),
                }
            )


def run_inference(
    model_key: str,
    checkpoint_path: Path,
    model_type: str,
    dataset_targets: Sequence[Tuple[str, Path, Dict[str, bool]]],
    args: argparse.Namespace,
    device: torch.device,
) -> List[Path]:
    model_output_root = args.outputs_dir / model_key
    model_output_root.mkdir(parents=True, exist_ok=True)

    model = load_model(checkpoint_path, model_type, device)
    produced_dirs: List[Path] = []

    for dataset_name, dataset_root, config in dataset_targets:
        dataset_output_dir = model_output_root / dataset_name
        dataset_output_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Inference {model_key} on {dataset_name}")
        transform = build_transform(
            args.load_size,
            args.crop_size,
            no_resize=config["no_resize"],
            no_crop=config["no_crop"],
        )

        try:
            dataset = BinaryImageDataset(dataset_root, transform)
        except ValueError as exc:
            print(f"[WARN] Skipping {dataset_name}: {exc}")
            continue

        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=safe_collate,
        )

        y_true_list: List[int] = []
        y_prob_list: List[float] = []
        paths: List[str] = []

        with torch.no_grad():
            for batch in loader:
                if batch is None:
                    continue
                images, labels, batch_paths = batch
                images = images.to(device)
                logits = model(images).view(-1)
                probs = torch.sigmoid(logits).cpu().numpy()
                y_prob_list.extend(probs.tolist())
                y_true_list.extend(labels.numpy().tolist())
                paths.extend(batch_paths)

        y_true = np.asarray(y_true_list, dtype=np.int64)
        y_prob = np.asarray(y_prob_list, dtype=np.float32)
        y_pred = (y_prob >= args.threshold).astype(np.int64)

        np.save(dataset_output_dir / "y_true.npy", y_true)
        np.save(dataset_output_dir / "y_pred.npy", y_pred)
        np.save(dataset_output_dir / "y_prob.npy", y_prob)
        np.save(dataset_output_dir / "class_names.npy", np.asarray(["0_real", "1_fake"], dtype=object))
        save_prediction_csv(dataset_output_dir, paths, y_true, y_pred, y_prob)
        produced_dirs.append(dataset_output_dir)

    return produced_dirs


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Sequence[str],
    title: str,
    output_path: Path,
    normalized: bool,
    dpi: int,
) -> None:
    plt.figure(figsize=(7.5, 6.5))
    fmt = ".2f" if normalized else "d"
    heatmap = sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap=sns.color_palette("Blues", as_cmap=True),
        cbar=True,
        square=True,
        linewidths=0.8,
        linecolor="white",
        annot_kws={"fontsize": 10, "fontweight": "semibold"},
    )
    heatmap.set_xticklabels(class_names, rotation=15, ha="right")
    heatmap.set_yticklabels(class_names, rotation=0)
    plt.xlabel("Predicted Label", fontweight="semibold")
    plt.ylabel("True Label", fontweight="semibold")
    plt.title(title, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()


def evaluate_predictions(
    model_key: str,
    dataset_output_dir: Path,
    args: argparse.Namespace,
) -> Optional[Dict[str, object]]:
    y_true_path = dataset_output_dir / "y_true.npy"
    y_pred_path = dataset_output_dir / "y_pred.npy"
    y_prob_path = dataset_output_dir / "y_prob.npy"
    if not y_true_path.is_file() or not y_pred_path.is_file() or not y_prob_path.is_file():
        print(f"[WARN] Skipping evaluation for {dataset_output_dir}: missing prediction arrays.")
        return None

    y_true = np.load(y_true_path)
    y_pred = np.load(y_pred_path)
    y_prob = np.load(y_prob_path)
    if len(y_true) == 0:
        print(f"[WARN] Skipping evaluation for {dataset_output_dir}: empty predictions.")
        return None

    dataset_name = dataset_output_dir.name
    accuracy = float(accuracy_score(y_true, y_pred))
    precision = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    recall = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    roc_auc = float(roc_auc_score(y_true, y_prob))

    cm_raw = confusion_matrix(y_true, y_pred, labels=[0, 1])
    row_sums = cm_raw.sum(axis=1, keepdims=True)
    cm_norm = np.divide(
        cm_raw.astype(np.float64),
        row_sums,
        out=np.zeros_like(cm_raw, dtype=np.float64),
        where=row_sums != 0,
    )

    eval_dir = dataset_output_dir / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    global_cm_dir = args.outputs_dir / "confusion_matrices"
    global_cm_dir.mkdir(parents=True, exist_ok=True)

    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["0_real", "1_fake"],
        digits=4,
        zero_division=0,
    )
    (eval_dir / "classification_report.txt").write_text(report + "\n", encoding="utf-8")

    raw_local = eval_dir / "confusion_matrix_raw.png"
    norm_local = eval_dir / "confusion_matrix_normalized.png"
    raw_global = global_cm_dir / f"{model_key}__{dataset_name}_raw.png"
    norm_global = global_cm_dir / f"{model_key}__{dataset_name}_normalized.png"
    plot_confusion_matrix(cm_raw, ["0_real", "1_fake"], f"{model_key} | {dataset_name} | Raw CM", raw_local, False, args.dpi)
    plot_confusion_matrix(cm_norm, ["0_real", "1_fake"], f"{model_key} | {dataset_name} | Normalized CM", norm_local, True, args.dpi)
    plot_confusion_matrix(cm_raw, ["0_real", "1_fake"], f"{model_key} | {dataset_name} | Raw CM", raw_global, False, args.dpi)
    plot_confusion_matrix(cm_norm, ["0_real", "1_fake"], f"{model_key} | {dataset_name} | Normalized CM", norm_global, True, args.dpi)

    metrics = {
        "model": model_key,
        "dataset": dataset_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
    }
    with (eval_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    return metrics


def write_results_csv(rows: Sequence[Dict[str, object]], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in SUMMARY_COLUMNS})


def plot_model_comparison(rows: Sequence[Dict[str, object]], output_path: Path, dpi: int) -> None:
    if not rows:
        return

    dataset_names = sorted({str(row["dataset"]) for row in rows})
    model_names = sorted({str(row["model"]) for row in rows})
    x = np.arange(len(dataset_names))
    width = min(0.8 / max(len(model_names), 1), 0.22)
    palette = sns.color_palette("deep", n_colors=len(model_names))

    fig, axes = plt.subplots(2, 2, figsize=(max(14, 1.6 * len(dataset_names)), 10), sharex=True)
    metric_axes = {
        "accuracy": axes[0, 0],
        "precision": axes[0, 1],
        "recall": axes[1, 0],
        "f1": axes[1, 1],
    }

    for metric, axis in metric_axes.items():
        for idx, model_name in enumerate(model_names):
            values = []
            for dataset_name in dataset_names:
                match = next(
                    (row for row in rows if row["model"] == model_name and row["dataset"] == dataset_name),
                    None,
                )
                values.append(match[metric] if match is not None else np.nan)
            offset = (idx - (len(model_names) - 1) / 2.0) * width
            axis.bar(
                x + offset,
                values,
                width=width,
                label=model_name,
                color=palette[idx],
                edgecolor="black",
                linewidth=0.5,
                alpha=0.9,
            )
        axis.set_title(metric.upper(), fontweight="bold")
        axis.set_ylim(0.0, 1.05)
        axis.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.5)

    for axis in axes[1]:
        axis.set_xticks(x)
        axis.set_xticklabels(dataset_names, rotation=25, ha="right")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=max(1, len(model_names)), frameon=True)
    fig.suptitle("Model Comparison Across Datasets", fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_pipeline_state(state_path: Path, state: Dict[str, Dict[str, str]]) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with state_path.open("w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2)


def load_pipeline_state(state_path: Path) -> Dict[str, Dict[str, str]]:
    if not state_path.is_file():
        return {}
    with state_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    args = parse_args()
    configure_plotting()

    datasets_root = resolve_datasets_root(args.datasets_root)
    args.datasets_root = datasets_root
    args.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    args.outputs_dir.mkdir(parents=True, exist_ok=True)

    state_path = args.outputs_dir / "pipeline_state.json"
    state = load_pipeline_state(state_path)
    state = normalize_state_with_existing_checkpoints(args, state)

    device = torch.device("cuda" if torch.cuda.is_available() and args.gpu_ids != "-1" else "cpu")
    dataset_targets = discover_dataset_targets(datasets_root, args.dataset_names)
    if not dataset_targets:
        print(f"[WARN] No binary-labeled evaluation datasets found under {datasets_root.resolve()}")

    if "train" in args.stages:
        for model_key in args.models:
            if model_key == "baseline":
                train_model("baseline", "baseline", args, datasets_root, state)
            elif model_key == "npr":
                train_model("npr", "npr", args, datasets_root, state)
        save_pipeline_state(state_path, state)

    if "finetune" in args.stages and "npr_finetune" in args.models:
        finetune_model(args, state)
        save_pipeline_state(state_path, state)

    evaluation_rows: List[Dict[str, object]] = []
    if "inference" in args.stages or "evaluate" in args.stages or "compare" in args.stages:
        for model_key in args.models:
            model_state = state.get(model_key)
            if not model_state:
                print(f"[WARN] No checkpoint metadata available for {model_key}; skipping.")
                continue

            checkpoint_path = Path(model_state["checkpoint"])
            if not checkpoint_path.is_file():
                print(f"[WARN] Checkpoint missing for {model_key}: {checkpoint_path}")
                continue

            if "inference" in args.stages:
                run_inference(
                    model_key=model_key,
                    checkpoint_path=checkpoint_path,
                    model_type=model_state["model_type"],
                dataset_targets=dataset_targets,
                args=args,
                device=device,
            )

            if "evaluate" in args.stages or "compare" in args.stages:
                model_output_root = args.outputs_dir / model_key
                for dataset_name, _, _ in dataset_targets:
                    result = evaluate_predictions(model_key, model_output_root / dataset_name, args)
                    if result is not None:
                        evaluation_rows.append(result)

    if "compare" in args.stages and evaluation_rows:
        results_csv = args.outputs_dir / "evaluation_results.csv"
        write_results_csv(evaluation_rows, results_csv)
        plot_model_comparison(evaluation_rows, args.outputs_dir / "comparison_plot.png", args.dpi)
        print(f"[INFO] Wrote summary CSV to {results_csv}")
        print(f"[INFO] Wrote comparison plot to {args.outputs_dir / 'comparison_plot.png'}")

    save_pipeline_state(state_path, state)


if __name__ == "__main__":
    main()
