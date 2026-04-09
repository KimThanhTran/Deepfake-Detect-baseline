import argparse
import os
import random
import sys
import time


def bootstrap_imports():
    print("Starting training script...")
    print("Importing PyTorch and project modules, please wait...", flush=True)

    import numpy as np
    import torch
    import torch.nn  # noqa: F401
    from PIL import Image  # noqa: F401
    from sklearn.metrics import roc_auc_score

    from data import create_dataloader
    from networks.trainer import Trainer
    from options.test_options import TestOptions
    from options.train_options import TrainOptions
    from util import Logger
    from validate import validate

    try:
        from tensorboardX import SummaryWriter
    except ImportError:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            class SummaryWriter:  # type: ignore[override]
                def __init__(self, *args, **kwargs):
                    pass

                def add_scalar(self, *args, **kwargs):
                    pass

                def close(self):
                    pass

    return {
        "np": np,
        "torch": torch,
        "create_dataloader": create_dataloader,
        "Trainer": Trainer,
        "TrainOptions": TrainOptions,
        "TestOptions": TestOptions,
        "Logger": Logger,
        "validate": validate,
        "SummaryWriter": SummaryWriter,
        "roc_auc_score": roc_auc_score,
    }


def seed_torch(seed, np, torch):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


def format_progress_bar(current, total, width=20):
    if total <= 0:
        total = 1
    ratio = min(max(current / total, 0.0), 1.0)
    filled = int(width * ratio)
    return "[{}{}]".format("=" * filled, "." * (width - filled))


def format_metric(value, precision=4):
    if value is None:
        return "N/A"
    return f"{float(value):.{precision}f}"


def compact_progress_line(epoch, total_epochs, batch_idx, total_batches, step, loss, lr):
    progress = format_progress_bar(batch_idx, total_batches, width=20)
    pct = (batch_idx / max(total_batches, 1)) * 100.0
    return (
        f"\rE{epoch:03d}/{total_epochs:03d} "
        f"B{batch_idx:03d}/{total_batches:03d} "
        f"{progress} {pct:5.1f}% "
        f"L:{loss:.4f} LR:{lr:.1e} S:{step}"
    )


def get_val_opt(TrainOptions):
    val_opt = TrainOptions().parse(print_options=False)
    val_opt.dataroot = "{}/{}/".format(val_opt.dataroot, val_opt.val_split)
    val_opt.isTrain = False
    val_opt.no_resize = False
    val_opt.no_crop = False
    val_opt.serial_batches = True
    return val_opt


def evaluate_validation_metrics(model_wrapper, data_loader, torch, roc_auc_score):
    model_wrapper.eval()
    losses = []
    y_true = []
    y_prob = []

    with torch.no_grad():
        for img, label in data_loader:
            in_tens = img.to(model_wrapper.device)
            targets = label.to(model_wrapper.device).float()
            logits = model_wrapper.model(in_tens).squeeze(1)
            loss = model_wrapper.loss_fn(logits, targets)
            probs = torch.sigmoid(logits)

            losses.append(loss.detach().item())
            y_true.extend(targets.cpu().numpy().tolist())
            y_prob.extend(probs.cpu().numpy().tolist())

    val_loss = float(sum(losses) / len(losses)) if losses else None
    val_auc = None
    if y_true and len(set(y_true)) > 1:
        try:
            val_auc = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            val_auc = None

    return val_loss, val_auc


def main():
    modules = bootstrap_imports()
    np = modules["np"]
    torch = modules["torch"]
    create_dataloader = modules["create_dataloader"]
    Trainer = modules["Trainer"]
    TrainOptions = modules["TrainOptions"]
    TestOptions = modules["TestOptions"]
    Logger = modules["Logger"]
    validate = modules["validate"]
    SummaryWriter = modules["SummaryWriter"]
    roc_auc_score = modules["roc_auc_score"]

    opt = TrainOptions().parse()
    seed_torch(100, np, torch)

    if opt.gpu_ids:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"Using GPU: {gpu_name} ({gpu_mem_gb:.1f} GB VRAM)")
        if opt.model_type == "baseline" and gpu_mem_gb <= 4.5 and opt.batch_size > 8:
            print(
                f"Reducing batch_size from {opt.batch_size} to 8 for baseline training on a {gpu_mem_gb:.1f} GB GPU."
            )
            opt.batch_size = 8

    test_dataroot = os.path.join(opt.dataroot, "test")
    opt.dataroot = "{}/{}/".format(opt.dataroot, opt.train_split)
    Logger(os.path.join(opt.checkpoints_dir, opt.name, "log.log"))
    print("  ".join(list(sys.argv)))

    val_opt = get_val_opt(TrainOptions)
    _ = TestOptions().parse(print_options=False)
    data_loader = create_dataloader(opt)
    val_loader = create_dataloader(val_opt)

    train_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "train"))
    val_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "val"))

    model = Trainer(opt)
    model.train()
    print(f"cwd: {os.getcwd()}")
    num_batches = len(data_loader)
    start_epoch = model.start_epoch if opt.continue_train else 0
    best_val_ap = model.resume_state.get("best_val_ap", float("-inf")) if opt.continue_train else float("-inf")
    best_val_acc = model.resume_state.get("best_val_acc", float("-inf")) if opt.continue_train else float("-inf")
    epochs_without_improvement = model.resume_state.get("epochs_without_improvement", 0) if opt.continue_train else 0

    if opt.continue_train:
        print(
            f"Resuming training from epoch {start_epoch + 1}/{opt.niter} "
            f"at step {model.total_steps} | best_ap={best_val_ap:.4f}"
        )

    for epoch in range(start_epoch, opt.niter):
        epoch_start_time = time.time()
        last_train_loss = None
        print(f"\nEpoch {epoch + 1}/{opt.niter}")

        for i, data in enumerate(data_loader):
            model.total_steps += 1
            model.set_input(data)
            model.optimize_parameters()
            last_train_loss = model.loss.detach().item()

            progress_line = compact_progress_line(
                epoch + 1,
                opt.niter,
                i + 1,
                num_batches,
                model.total_steps,
                last_train_loss,
                model.lr,
            )
            print(progress_line, end="", flush=True)

            if model.total_steps % opt.loss_freq == 0:
                print()
                print(
                    time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()),
                    "Train loss: {} at step: {} lr {}".format(model.loss, model.total_steps, model.lr),
                )
                train_writer.add_scalar("loss", model.loss, model.total_steps)

            if model.total_steps % opt.save_latest_freq == 0:
                model.start_epoch = epoch
                model.save_networks(
                    "latest",
                    extra_state={
                        "best_val_ap": best_val_ap,
                        "best_val_acc": best_val_acc,
                        "epochs_without_improvement": epochs_without_improvement,
                    },
                )

        epoch_time = time.time() - epoch_start_time
        print()

        if epoch % opt.delr_freq == 0 and epoch != 0:
            print(
                time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()),
                "changing lr at the end of epoch %d, iters %d" % (epoch, model.total_steps),
            )
            model.adjust_learning_rate()

        model.eval()
        val_loss, val_auc = evaluate_validation_metrics(model, val_loader, torch, roc_auc_score)
        acc, ap = validate(model.model, val_opt)[:2]
        train_writer.add_scalar("epoch_loss", last_train_loss, model.total_steps)
        val_writer.add_scalar("loss", val_loss, model.total_steps)
        val_writer.add_scalar("accuracy", acc, model.total_steps)
        val_writer.add_scalar("ap", ap, model.total_steps)
        if val_auc is not None:
            val_writer.add_scalar("auc", val_auc, model.total_steps)
        epoch_summary = (
            f"[Epoch {epoch + 1:04d}/{opt.niter:04d}] "
            f"TRAIN loss={format_metric(last_train_loss)} lr={model.lr:.6f} time={epoch_time:.2f}s || "
            f"VAL loss={format_metric(val_loss)} acc={format_metric(acc)} ap={format_metric(ap)} "
            f"auc={format_metric(val_auc)}"
        )
        print("-" * len(epoch_summary))
        print(epoch_summary)
        print("-" * len(epoch_summary))

        improved = ap > best_val_ap
        if improved:
            best_val_ap = ap
            best_val_acc = acc
            epochs_without_improvement = 0
            model.start_epoch = epoch + 1
            model.save_networks(
                "best",
                extra_state={
                    "best_val_ap": best_val_ap,
                    "best_val_acc": best_val_acc,
                    "epochs_without_improvement": epochs_without_improvement,
                },
            )
            print(
                f"New best model saved | val_ap={best_val_ap:.4f} | "
                f"val_acc={best_val_acc:.4f}"
            )
        else:
            epochs_without_improvement += 1
            print(
                f"Early-stop counter: {epochs_without_improvement}/{opt.earlystop_epoch} "
                f"| best_ap={best_val_ap:.4f}"
            )

        if epochs_without_improvement >= opt.earlystop_epoch:
            print(
                f"Early stopping triggered after {epoch + 1} epochs "
                f"without val_ap improvement for {opt.earlystop_epoch} consecutive epochs."
            )
            break

        model.start_epoch = epoch + 1
        model.save_networks(
            "latest",
            extra_state={
                "best_val_ap": best_val_ap,
                "best_val_acc": best_val_acc,
                "epochs_without_improvement": epochs_without_improvement,
            },
        )

        model.train()

    train_writer.close()
    val_writer.close()
    model.start_epoch = epoch + 1
    model.save_networks(
        "last",
        extra_state={
            "best_val_ap": best_val_ap,
            "best_val_acc": best_val_acc,
            "epochs_without_improvement": epochs_without_improvement,
        },
    )


if __name__ == "__main__":
    main()
