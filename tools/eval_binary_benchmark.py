import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from networks import build_detector


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a binary benchmark root recursively.")
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--model_path", required=True, type=Path)
    parser.add_argument("--model_type", default="baseline", choices=["baseline", "npr", "hybrid"])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--load_size", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output_json", type=Path, default=None)
    return parser.parse_args()


def collect_samples(root: Path):
    direct_real = root / "0_real"
    direct_fake = root / "1_fake"
    if direct_real.is_dir() and direct_fake.is_dir():
        samples = []
        for folder, label in ((direct_real, 0), (direct_fake, 1)):
            for path in sorted(folder.iterdir()):
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                    samples.append((path, label))
        return samples

    samples = []
    for child in sorted(root.iterdir()):
        if child.is_dir():
            samples.extend(collect_samples(child))
    return samples


class ImageDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")
        return self.transform(image), label


def load_model(model_path: Path, model_type: str, device: torch.device):
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "model_type" in checkpoint:
        model_type = checkpoint["model_type"]
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    first_key = next(iter(state_dict))
    if first_key.startswith("module."):
        state_dict = {key.replace("module.", "", 1): value for key, value in state_dict.items()}

    model = build_detector(model_type=model_type, num_classes=1)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def main():
    args = parse_args()
    samples = collect_samples(args.root)
    if not samples:
        raise ValueError(f"No binary samples found under {args.root}")

    transform = transforms.Compose(
        [
            transforms.Resize((args.load_size, args.load_size)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )

    dataset = ImageDataset(samples, transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model_path, args.model_type, device)

    y_true = []
    y_prob = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images).view(-1)
            probs = torch.sigmoid(logits).cpu().numpy()
            y_prob.extend(probs.tolist())
            y_true.extend(labels.numpy().tolist())

    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float32)
    y_pred = (y_prob >= args.threshold).astype(np.int64)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    real_mask = y_true == 0
    fake_mask = y_true == 1

    result = {
        "root": str(args.root.resolve()),
        "num_samples": int(len(y_true)),
        "accuracy_pct": round(accuracy_score(y_true, y_pred) * 100, 2),
        "real_accuracy_pct": round(accuracy_score(y_true[real_mask], y_pred[real_mask]) * 100, 2),
        "fake_accuracy_pct": round(accuracy_score(y_true[fake_mask], y_pred[fake_mask]) * 100, 2),
        "ap_pct": round(average_precision_score(y_true, y_prob) * 100, 2),
        "auc_pct": round(roc_auc_score(y_true, y_prob) * 100, 2),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    text = json.dumps(result, indent=2)
    print(text)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
