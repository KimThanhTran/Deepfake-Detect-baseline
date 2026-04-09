import argparse
import csv
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from networks import build_detector


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

BENCHMARKS = {
    "ForenSynths": {
        "root": "dataset/ForenSynths/test",
        "no_resize": False,
        "no_crop": True,
    },
    "GANGen-Detection": {
        "root": "dataset/GANGen-Detection",
        "no_resize": True,
        "no_crop": True,
    },
    "DiffusionForensics": {
        "root": "dataset/DiffusionForensics",
        "no_resize": False,
        "no_crop": True,
    },
    "UniversalFakeDetect": {
        "root": "dataset/UniversalFakeDetect",
        "no_resize": False,
        "no_crop": True,
    },
    "Diffusion1kStep": {
        "root": "dataset/Diffusion1kStep",
        "no_resize": False,
        "no_crop": True,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate NPR across benchmarks with full metrics and Grad-CAM XAI."
    )
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--load_size", type=int, default=256)
    parser.add_argument("--crop_size", type=int, default=224)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--benchmark", nargs="*", default=list(BENCHMARKS.keys()))
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--xai_samples_per_class", type=int, default=2)
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--model_type", default="npr", choices=["baseline", "npr"])
    return parser.parse_args()


def list_images(folder):
    images = []
    for path in sorted(Path(folder).iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(path)
    return images


def collect_labeled_samples(root):
    root = Path(root)
    direct_real = root / "0_real"
    direct_fake = root / "1_fake"
    if direct_real.is_dir() and direct_fake.is_dir():
        samples = []
        for image_path in list_images(direct_real):
            samples.append((image_path, 0))
        for image_path in list_images(direct_fake):
            samples.append((image_path, 1))
        return samples

    samples = []
    for child in sorted(root.iterdir()):
        if child.is_dir():
            samples.extend(collect_labeled_samples(child))
    return samples


def build_transform(no_resize, no_crop, load_size, crop_size):
    if no_crop:
        crop_transform = transforms.Lambda(lambda img: img)
    else:
        crop_transform = transforms.CenterCrop(crop_size)

    if no_resize:
        resize_transform = transforms.Lambda(lambda img: img)
    else:
        resize_transform = transforms.Resize((load_size, load_size))

    return transforms.Compose(
        [
            resize_transform,
            crop_transform,
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
        ]
    )


class PathImageDataset(Dataset):
    def __init__(self, root, transform):
        self.root = Path(root)
        self.samples = collect_labeled_samples(root)
        self.transform = transform
        if not self.samples:
            raise ValueError(f"No labeled images found under {root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        try:
            image = Image.open(path).convert("RGB")
            tensor = self.transform(image)
            return tensor, label, str(path)
        except Exception as exc:
            print(f"Skipping unreadable image: {path} ({exc})")
            return None


def safe_collate(batch):
    valid_items = [item for item in batch if item is not None]
    if not valid_items:
        return None

    images = torch.stack([item[0] for item in valid_items], dim=0)
    labels = torch.tensor([item[1] for item in valid_items], dtype=torch.long)
    paths = [item[2] for item in valid_items]
    return images, labels, paths


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.forward_handle = target_layer.register_forward_hook(self._save_activation)
        self.backward_handle = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inputs, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor):
        self.model.zero_grad(set_to_none=True)
        output = self.model(input_tensor)
        score = output.view(-1)[0]
        score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(
            cam, size=input_tensor.shape[-2:], mode="bilinear", align_corners=False
        )
        cam = cam[0, 0].cpu().numpy()
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)
        return cam

    def close(self):
        self.forward_handle.remove()
        self.backward_handle.remove()


def load_model(model_path, device, model_type="npr"):
    checkpoint = torch.load(model_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_type" in checkpoint:
        model_type = checkpoint["model_type"]
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    if list(state_dict.keys())[0].startswith("module."):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

    model = build_detector(model_type=model_type, num_classes=1)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def compute_metrics(y_true, y_score, threshold):
    y_pred = (y_score >= threshold).astype(np.int32)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    real_mask = y_true == 0
    fake_mask = y_true == 1

    metrics = {
        "num_samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "average_precision": float(average_precision_score(y_true, y_score)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "real_accuracy": float(accuracy_score(y_true[real_mask], y_pred[real_mask])),
        "fake_accuracy": float(accuracy_score(y_true[fake_mask], y_pred[fake_mask])),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    return metrics, y_pred


def tensor_to_bgr_image(tensor):
    image = tensor.detach().cpu().numpy().transpose(1, 2, 0)
    image = image * np.array(STD) + np.array(MEAN)
    image = np.clip(image, 0, 1)
    image = (image * 255).astype(np.uint8)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def load_tensor_from_path(path, transform):
    image = Image.open(path).convert("RGB")
    return transform(image)


def build_overlay(image_bgr, cam):
    heatmap = np.uint8(255 * cam)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image_bgr, 0.55, heatmap, 0.45, 0)
    panel = cv2.hconcat([image_bgr, heatmap, overlay])
    return panel


def choose_xai_indices(records, samples_per_class):
    selected = []
    for label in [0, 1]:
        matching = [idx for idx, record in enumerate(records) if record["label"] == label]
        selected.extend(matching[:samples_per_class])
    return selected


def evaluate_dataset(model, device, dataset_name, dataset_root, config, args, run_dir):
    transform = build_transform(
        no_resize=config["no_resize"],
        no_crop=config["no_crop"],
        load_size=args.load_size,
        crop_size=args.crop_size,
    )
    dataset = PathImageDataset(dataset_root, transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=safe_collate,
    )

    y_true = []
    y_score = []
    paths = []

    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            images, labels, batch_paths = batch
            images = images.to(device)
            logits = model(images).view(-1)
            probs = torch.sigmoid(logits).cpu().numpy()
            y_score.extend(probs.tolist())
            y_true.extend(labels.numpy().tolist())
            paths.extend(batch_paths)

    y_true = np.array(y_true)
    y_score = np.array(y_score)
    metrics, y_pred = compute_metrics(y_true, y_score, args.threshold)

    records = []
    for path, label, score, pred in zip(paths, y_true, y_score, y_pred):
        records.append(
            {
                "path": path,
                "label": int(label),
                "score_fake": float(score),
                "prediction": int(pred),
                "correct": int(label == pred),
            }
        )

    dataset_dir = Path(run_dir) / dataset_name
    xai_dir = dataset_dir / "xai"
    xai_dir.mkdir(parents=True, exist_ok=True)

    grad_cam = GradCAM(model, model.layer2[-1])
    selected_indices = choose_xai_indices(records, args.xai_samples_per_class)

    for sample_id, sample_index in enumerate(selected_indices):
        path = records[sample_index]["path"]
        label = records[sample_index]["label"]
        tensor = load_tensor_from_path(path, transform)
        input_tensor = tensor.unsqueeze(0).to(device)
        cam = grad_cam.generate(input_tensor)
        image_bgr = tensor_to_bgr_image(tensor)
        panel = build_overlay(image_bgr, cam)

        prediction = int(records[sample_index]["prediction"])
        score = records[sample_index]["score_fake"]
        file_stem = Path(path).stem
        output_name = (
            f"{sample_id:02d}_label{label}_pred{prediction}_score{score:.4f}_{file_stem}.jpg"
        )
        cv2.imwrite(str(xai_dir / output_name), panel)

    grad_cam.close()

    predictions_path = dataset_dir / "predictions.csv"
    with predictions_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["path", "label", "score_fake", "prediction", "correct"],
        )
        writer.writeheader()
        writer.writerows(records)

    metrics["dataset"] = dataset_name
    metrics["dataset_root"] = str(Path(dataset_root).resolve())
    return metrics


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    run_dir = Path(args.output_dir) / f"benchmark_eval_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(args.model_path, device, model_type=args.model_type)
    summary = []

    def write_summary_files():
        ordered_summary = sorted(summary, key=lambda item: (item["benchmark"], item["subset"]))

        summary_csv = run_dir / "summary.csv"
        with summary_csv.open("w", newline="", encoding="utf-8") as handle:
            fieldnames = [
                "dataset",
                "benchmark",
                "subset",
                "num_samples",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "average_precision",
                "roc_auc",
                "real_accuracy",
                "fake_accuracy",
                "tn",
                "fp",
                "fn",
                "tp",
                "dataset_root",
            ]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(ordered_summary)

        with (run_dir / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "device": str(device),
                    "model_path": str(Path(args.model_path).resolve()),
                    "threshold": args.threshold,
                    "generated_at": timestamp,
                    "results": ordered_summary,
                },
                handle,
                indent=2,
            )

    for benchmark_name in args.benchmark:
        config = BENCHMARKS[benchmark_name]
        benchmark_root = Path(config["root"])
        for dataset_dir in sorted(benchmark_root.iterdir()):
            if not dataset_dir.is_dir():
                continue
            full_name = f"{benchmark_name}/{dataset_dir.name}"
            print(f"Evaluating {full_name} on {device}...")
            try:
                metrics = evaluate_dataset(
                    model=model,
                    device=device,
                    dataset_name=full_name.replace("/", "__"),
                    dataset_root=str(dataset_dir),
                    config=config,
                    args=args,
                    run_dir=run_dir,
                )
            except ValueError as exc:
                print(f"  skipped: {exc}")
                continue
            metrics["benchmark"] = benchmark_name
            metrics["subset"] = dataset_dir.name
            summary.append(metrics)
            write_summary_files()
            print(
                f"  acc={metrics['accuracy']:.4f} ap={metrics['average_precision']:.4f} "
                f"f1={metrics['f1']:.4f} auc={metrics['roc_auc']:.4f}"
            )
    write_summary_files()

    print(f"Done. Results saved to {run_dir.resolve()}")


if __name__ == "__main__":
    main()
