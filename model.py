"""
On-Device Model Compression Pipeline

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - load_fashion_mnist
import os
import gzip
import tempfile
import urllib.request
import numpy as np
import torch

def load_fashion_mnist(n_train=6000, n_test=1000):
    base_url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/"
    cache_dir = tempfile.gettempdir()

    filenames = {
        "train_images": "train-images-idx3-ubyte.gz",
        "train_labels": "train-labels-idx1-ubyte.gz",
        "test_images": "t10k-images-idx3-ubyte.gz",
        "test_labels": "t10k-labels-idx1-ubyte.gz",
    }

    paths = {}

    # Download each file only if it is not already present.
    for key, filename in filenames.items():
        path = os.path.join(cache_dir, filename)
        paths[key] = path

        if not os.path.exists(path):
            urllib.request.urlretrieve(base_url + filename, path)

    # Parse image IDX files.
    def load_images(path, n):
        with gzip.open(path, "rb") as f:
            raw = f.read()

        images = np.frombuffer(
            raw,
            dtype=np.uint8,
            offset=16
        ).reshape(-1, 1, 28, 28)

        images = images[:n]

        return torch.from_numpy(
            images.astype(np.float32) / 255.0
        )

    # Parse label IDX files.
    def load_labels(path, n):
        with gzip.open(path, "rb") as f:
            raw = f.read()

        labels = np.frombuffer(
            raw,
            dtype=np.uint8,
            offset=8
        )[:n]

        return torch.from_numpy(
            labels.astype(np.int64)
        )

    X_train = load_images(paths["train_images"], n_train)
    y_train = load_labels(paths["train_labels"], n_train)

    X_test = load_images(paths["test_images"], n_test)
    y_test = load_labels(paths["test_labels"], n_test)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
    }

# Step 2 - SmallCNN
import torch
import torch.nn as nn
import torch.nn.functional as F

class SmallCNN(nn.Module):

    def __init__(self, c1=16, c2=32, hidden=64, n_classes=10):
        super().__init__()

        # Store network widths so a student can be built
        # from a teacher's configuration.
        self.c1, self.c2, self.hidden = c1, c2, hidden

        self.conv1 = nn.Conv2d(1, c1, 3, padding=1)
        self.conv2 = nn.Conv2d(c1, c2, 3, padding=1)

        self.fc1 = nn.Linear(c2 * 7 * 7, hidden)
        self.fc2 = nn.Linear(hidden, n_classes)

    def forward(self, x):
        # First convolution -> ReLU -> 2x2 max pooling
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)

        # Second convolution -> ReLU -> 2x2 max pooling
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)

        # Flatten while preserving the batch dimension
        x = torch.flatten(x, 1)

        # Fully connected layer -> ReLU -> output logits
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

# Step 3 - train_classifier
def train_classifier(
    model,
    X,
    y,
    epochs=1,
    lr=1e-3,
    batch_size=64,
    seed=0
):
    # Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Generator used to produce reproducible shuffles
    generator = torch.Generator().manual_seed(seed)

    model.train()
    epoch_losses = []

    for _ in range(epochs):
        # Seeded shuffled indices for this epoch
        indices = torch.randperm(X.shape[0], generator=generator)

        total_loss = 0.0
        total_samples = 0

        for start in range(0, X.shape[0], batch_size):
            batch_idx = indices[start:start + batch_size]

            xb = X[batch_idx]
            yb = y[batch_idx]

            optimizer.zero_grad()

            logits = model(xb)
            loss = F.cross_entropy(logits, yb)

            loss.backward()
            optimizer.step()

            batch_size_actual = yb.shape[0]
            total_loss += loss.item() * batch_size_actual
            total_samples += batch_size_actual

        # Mean loss over all samples in the epoch
        epoch_losses.append(total_loss / total_samples)

    return epoch_losses


def accuracy(model, X, y, batch_size=256):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for start in range(0, X.shape[0], batch_size):
            xb = X[start:start + batch_size]
            yb = y[start:start + batch_size]

            logits = model(xb)
            predictions = logits.argmax(dim=1)

            correct += (predictions == yb).sum().item()
            total += yb.numel()

    return float(correct / total)

# Step 4 - model_size_mb
def model_size_mb(model, bits=32):
    # Total number of parameters, assuming every parameter
    # is stored using the specified number of bits.
    n_params = sum(p.numel() for p in model.parameters())

    size_mb = n_params * bits / 8 / 1e6

    return round(size_mb, 4)

def count_params(model):
    total = 0
    nonzero = 0

    for p in model.parameters():
        total += p.numel()
        nonzero += torch.count_nonzero(p).item()

    return int(total), int(nonzero)

def sparsity(model):
    total, nonzero = count_params(model)

    if total == 0:
        return 0.0

    return round(1.0 - nonzero / total, 4)

# Step 5 - measure_latency_ms
import time

def measure_latency_ms(
    model,
    input_shape=(1, 1, 28, 28),
    iters=20,
    warmup=5
):
    # Run inference on CPU.
    model = model.cpu()
    model.eval()

    # Single-example random input.
    x = torch.randn(input_shape, device="cpu")

    with torch.no_grad():
        # Warmup passes are not timed.
        for _ in range(warmup):
            model(x)

        # Time individual forward passes.
        times = []

        for _ in range(iters):
            start = time.perf_counter()
            model(x)
            end = time.perf_counter()

            times.append((end - start) * 1000.0)

    # Median latency in milliseconds per pass.
    return float(torch.tensor(times).median().item())

def fps_from_latency(latency_ms):
    return round(1000.0 / latency_ms, 1)

# Step 6 - global_magnitude_prune
def global_magnitude_prune(model, target_sparsity):
    # Collect the names and weight tensors of all Conv2d/Linear layers.
    weights = []

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            weights.append((f"{name}.weight", module.weight))

    # Handle the case where there are no prunable weights.
    if not weights:
        return {}

    # Concatenate all absolute weight magnitudes into one global vector.
    magnitudes = torch.cat([
        weight.detach().abs().reshape(-1)
        for _, weight in weights
    ])

    total_weights = magnitudes.numel()

    # Number of weights to prune.
    k = int(torch.floor(
        torch.tensor(target_sparsity * total_weights)
    ).item())

    # Clamp to a valid range.
    k = max(0, min(k, total_weights))

    # Build masks. True means the weight is kept.
    masks = {}

    with torch.no_grad():
        if k == 0:
            # No pruning.
            for name, weight in weights:
                masks[name] = torch.ones_like(
                    weight, dtype=torch.bool
                )
            return masks

        if k == total_weights:
            # Prune every weight.
            for name, weight in weights:
                weight.zero_()
                masks[name] = torch.zeros_like(
                    weight, dtype=torch.bool
                )
            return masks

        # kthvalue is 1-indexed. We use k+1 so that k entries
        # are strictly below the threshold.
        threshold = torch.kthvalue(
            magnitudes,
            k + 1
        ).values

        for name, weight in weights:
            mask = weight.detach().abs() >= threshold

            # Zero the pruned weights in place.
            weight.masked_fill_(~mask, 0.0)

            masks[name] = mask.clone()

    return masks

def weight_sparsity(masks):
    total = 0
    false_count = 0

    for mask in masks.values():
        total += mask.numel()
        false_count += (~mask).sum().item()

    if total == 0:
        return 0.0

    return round(false_count / total, 4)

# Step 7 - fine_tune_pruned
def apply_masks(model, masks):
    # Reapply each pruning mask directly to the corresponding
    # parameter so all pruned weights remain exactly zero.
    with torch.no_grad():
        named_parameters = dict(model.named_parameters())

        for name, mask in masks.items():
            if name in named_parameters:
                named_parameters[name].mul_(mask.to(
                    device=named_parameters[name].device,
                    dtype=named_parameters[name].dtype
                ))

def fine_tune_pruned(
    model,
    masks,
    X,
    y,
    epochs=1,
    lr=5e-4,
    batch_size=64,
    seed=0
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Reproducible shuffled mini-batches.
    generator = torch.Generator().manual_seed(seed)

    # Make sure the model starts with the masks applied.
    apply_masks(model, masks)

    model.train()
    epoch_losses = []

    for _ in range(epochs):
        indices = torch.randperm(
            X.shape[0],
            generator=generator
        )

        total_loss = 0.0
        total_samples = 0

        for start in range(0, X.shape[0], batch_size):
            batch_idx = indices[start:start + batch_size]

            xb = X[batch_idx]
            yb = y[batch_idx]

            optimizer.zero_grad()

            logits = model(xb)
            loss = F.cross_entropy(logits, yb)

            loss.backward()
            optimizer.step()

            # Restore the pruning mask immediately after every
            # optimizer update so pruned weights stay exactly zero.
            apply_masks(model, masks)

            batch_n = yb.shape[0]
            total_loss += loss.item() * batch_n
            total_samples += batch_n

        epoch_losses.append(total_loss / total_samples)

    return epoch_losses

# Step 8 - iterative_prune
def prune_schedule(target, n_rounds):
    if n_rounds <= 0:
        return []

    schedule = []

    for k in range(1, n_rounds + 1):
        sparsity = target * (1.0 - (1.0 - k / n_rounds) ** 3)
        schedule.append(round(sparsity, 4))

    return schedule

def iterative_prune(
    model,
    X,
    y,
    target,
    n_rounds,
    epochs_per_round=1,
    lr=5e-4,
    seed=0
):
    schedule = prune_schedule(target, n_rounds)

    masks = {}
    history = []

    for round_idx, scheduled_sparsity in enumerate(schedule):
        # Prune the current model to the cumulative sparsity
        # specified by this round.
        masks = global_magnitude_prune(
            model,
            scheduled_sparsity
        )

        # Fine-tune while preserving all currently pruned weights.
        losses = fine_tune_pruned(
            model,
            masks,
            X,
            y,
            epochs=epochs_per_round,
            lr=lr,
            batch_size=64,
            seed=seed + round_idx
        )

        # The required history loss is the last epoch's mean loss.
        train_loss = losses[-1] if losses else 0.0

        history.append(
            (
                weight_sparsity(masks),
                float(train_loss)
            )
        )

    return masks, history

# Step 9 - sparse_storage_mb
def sparse_storage_mb(model, masks, value_bits=32, index_bits=16):
    total_bits = 0

    # Parameters included in the pruning masks are stored in
    # compressed-sparse form.
    named_parameters = dict(model.named_parameters())

    for name, mask in masks.items():
        if name not in named_parameters:
            continue

        nonzero = int(mask.sum().item())

        total_bits += nonzero * (value_bits + index_bits)

    # Parameters without masks (e.g. biases) remain dense.
    masked_names = set(masks.keys())

    for name, param in model.named_parameters():
        if name not in masked_names:
            total_bits += param.numel() * value_bits

    return round(total_bits / 8e6, 4)

def pruning_breakeven_sparsity(value_bits=32, index_bits=16):
    sparsity = 1.0 - value_bits / (value_bits + index_bits)
    return round(sparsity, 4)

# Step 10 - quantize_symmetric
def quantize_symmetric(w, bits):
    qmax = 2 ** (bits - 1) - 1

    abs_max = w.abs().max()

    # Use scale = 1.0 for an all-zero tensor.
    if abs_max.item() == 0:
        scale = 1.0
    else:
        scale = abs_max.item() / qmax

    q = torch.round(w / scale)
    q = torch.clamp(q, -qmax, qmax).to(torch.int32)

    return q, float(scale)

def dequantize(q, scale):
    return q.float() * scale

def fake_quantize(w, bits):
    q, scale = quantize_symmetric(w, bits)
    return dequantize(q, scale)

# Step 11 - quantize_per_channel
def quantize_per_channel(w, bits):
    qmax = 2 ** (bits - 1) - 1

    # Flatten everything except dimension 0 so each output channel
    # gets a single scale based on its maximum absolute value.
    channel_absmax = w.abs().reshape(w.shape[0], -1).max(dim=1).values

    # Use scale = 1.0 for channels whose weights are all zero.
    scales = torch.where(
        channel_absmax == 0,
        torch.ones_like(channel_absmax),
        channel_absmax / qmax
    )

    # Reshape scales so they broadcast across all remaining dimensions.
    scale_shape = [w.shape[0]] + [1] * (w.ndim - 1)
    scales_broadcast = scales.reshape(scale_shape)

    q = torch.round(w / scales_broadcast)
    q = torch.clamp(q, -qmax, qmax).to(torch.int32)

    return q, scales.float()

def fake_quantize_per_channel(w, bits):
    q, scales = quantize_per_channel(w, bits)

    scale_shape = [w.shape[0]] + [1] * (w.ndim - 1)
    scales_broadcast = scales.reshape(scale_shape)

    return q.float() * scales_broadcast

def quantization_mse(w, w_hat):
    return float(torch.mean((w - w_hat) ** 2).item())

# Step 12 - quantize_model
import copy

def quantize_model(model, bits, per_channel=True):
    # Deep copy so the original model remains unchanged.
    qmodel = copy.deepcopy(model)

    scales = {}

    with torch.no_grad():
        for name, module in qmodel.named_modules():
            if not isinstance(module, (nn.Conv2d, nn.Linear)):
                continue

            if per_channel:
                q, scale = quantize_per_channel(module.weight, bits)
                scale_for_weight = scale.reshape(
                    [scale.shape[0]] + [1] * (module.weight.ndim - 1)
                )
                module.weight.copy_(
                    q.float() * scale_for_weight
                )
            else:
                q, scale = quantize_symmetric(
                    module.weight, bits
                )
                module.weight.copy_(
                    q.float() * scale
                )

            weight_name = f"{name}.weight"
            scales[weight_name] = scale

    return qmodel, scales

def quantized_size_mb(model, bits):
    total_bits = 0

    for name, param in model.named_parameters():
        # Conv2d/Linear weights are stored at the requested bit width.
        if name.endswith(".weight"):
            total_bits += param.numel() * bits

        # Biases and any other non-weight parameters remain float32.
        else:
            total_bits += param.numel() * 32

    return round(total_bits / 8e6, 4)

# Step 13 - quantization_snr_db
import math

def quantization_snr_db(w, bits, per_channel=False):
    if per_channel:
        w_hat = fake_quantize_per_channel(w, bits)
    else:
        w_hat = fake_quantize(w, bits)

    signal_power = torch.mean(w ** 2).item()
    noise_power = torch.mean((w - w_hat) ** 2).item()

    if noise_power == 0:
        return float("inf")

    if signal_power == 0:
        return float("-inf")

    snr_db = 10.0 * math.log10(signal_power / noise_power)

    return round(snr_db, 2)

def snr_per_bit(w, bits_list):
    return {
        bits: quantization_snr_db(w, bits)
        for bits in bits_list
    }

def db_gain_per_bit(snr_table):
    if len(snr_table) < 2:
        return 0.0

    sorted_bits = sorted(snr_table)

    min_bits = sorted_bits[0]
    max_bits = sorted_bits[-1]

    if max_bits == min_bits:
        return 0.0

    gain = (
        snr_table[max_bits] - snr_table[min_bits]
    ) / (max_bits - min_bits)

    return round(gain, 2)

# Step 14 - activation_ranges
def activation_ranges(model, X, batch_size=256):
    ranges = {}
    hooks = []

    # Register a forward hook on every Conv2d and Linear layer.
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):

            def hook_fn(module, inputs, output, layer_name=name):
                absmax = output.detach().abs().max().item()

                if layer_name not in ranges:
                    ranges[layer_name] = float(absmax)
                else:
                    ranges[layer_name] = max(
                        ranges[layer_name],
                        float(absmax)
                    )

            hooks.append(module.register_forward_hook(hook_fn))

    model.eval()

    try:
        with torch.no_grad():
            for start in range(0, X.shape[0], batch_size):
                xb = X[start:start + batch_size]
                model(xb)
    finally:
        # Always remove hooks, even if the forward pass raises an error.
        for hook in hooks:
            hook.remove()

    return ranges

def activation_scales(ranges, bits):
    qmax = 2 ** (bits - 1) - 1

    return {
        name: absmax / qmax
        for name, absmax in ranges.items()
    }

# Step 15 - distillation_loss
def soft_targets(logits, T):
    return F.softmax(logits / T, dim=1)

def distillation_loss(
    student_logits,
    teacher_logits,
    y,
    T=4.0,
    alpha=0.7
):
    # When alpha == 0, return plain cross-entropy exactly.
    if alpha == 0.0:
        return F.cross_entropy(student_logits, y)

    # If student and teacher logits are exactly identical,
    # the KL divergence is mathematically zero.
    # student_logits.sum() * 0 keeps the result differentiable.
    if torch.equal(student_logits, teacher_logits):
        kl_loss = student_logits.sum() * 0.0
    else:
        student_log_probs = F.log_softmax(
            student_logits / T,
            dim=1
        )

        teacher_probs = F.softmax(
            teacher_logits / T,
            dim=1
        )

        kl_loss = F.kl_div(
            student_log_probs,
            teacher_probs,
            reduction="batchmean"
        )

    ce_loss = F.cross_entropy(student_logits, y)

    return (
        alpha * (T ** 2) * kl_loss
        + (1.0 - alpha) * ce_loss
    )

# Step 16 - train_student
def train_student(
    student,
    teacher,
    X,
    y,
    epochs=1,
    lr=1e-3,
    batch_size=64,
    T=4.0,
    alpha=0.7,
    seed=0
):
    optimizer = torch.optim.Adam(student.parameters(), lr=lr)

    # Reproducible shuffled mini-batches.
    generator = torch.Generator().manual_seed(seed)

    # The teacher is frozen and used only for inference.
    teacher.eval()
    student.train()

    epoch_losses = []

    for _ in range(epochs):
        indices = torch.randperm(
            X.shape[0],
            generator=generator
        )

        total_loss = 0.0
        total_samples = 0

        for start in range(0, X.shape[0], batch_size):
            batch_idx = indices[start:start + batch_size]

            xb = X[batch_idx]
            yb = y[batch_idx]

            # Produce teacher logits without tracking gradients.
            with torch.no_grad():
                teacher_logits = teacher(xb)

            student_logits = student(xb)

            loss = distillation_loss(
                student_logits,
                teacher_logits,
                yb,
                T=T,
                alpha=alpha
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_n = yb.shape[0]
            total_loss += loss.item() * batch_n
            total_samples += batch_n

        epoch_losses.append(total_loss / total_samples)

    return epoch_losses

# Step 17 - compare_students
def compare_students(
    teacher,
    X_train,
    y_train,
    X_test,
    y_test,
    c1=8,
    c2=16,
    hidden=32,
    epochs=2,
    seed=0
):
    # Build the two students from exactly the same initialization.
    torch.manual_seed(seed)
    student_kd = SmallCNN(
        c1=c1,
        c2=c2,
        hidden=hidden
    )

    torch.manual_seed(seed)
    student_plain = SmallCNN(
        c1=c1,
        c2=c2,
        hidden=hidden
    )

    # Train one student with knowledge distillation.
    train_student(
        student_kd,
        teacher,
        X_train,
        y_train,
        epochs=epochs,
        seed=seed
    )

    # Train the other student using ordinary cross-entropy.
    train_classifier(
        student_plain,
        X_train,
        y_train,
        epochs=epochs,
        seed=seed
    )

    # Gather parameter counts.
    teacher_params = sum(
        p.numel() for p in teacher.parameters()
    )
    kd_params = sum(
        p.numel() for p in student_kd.parameters()
    )
    plain_params = sum(
        p.numel() for p in student_plain.parameters()
    )

    # Measure test accuracy.
    teacher_accuracy = round(
        accuracy(teacher, X_test, y_test),
        4
    )
    kd_accuracy = round(
        accuracy(student_kd, X_test, y_test),
        4
    )
    plain_accuracy = round(
        accuracy(student_plain, X_test, y_test),
        4
    )

    result = {
        "teacher": {
            "accuracy": teacher_accuracy,
            "params": int(teacher_params),
            "size_mb": model_size_mb(teacher)
        },
        "student_kd": {
            "accuracy": kd_accuracy,
            "params": int(kd_params),
            "size_mb": model_size_mb(student_kd)
        },
        "student_plain": {
            "accuracy": plain_accuracy,
            "params": int(plain_params),
            "size_mb": model_size_mb(student_plain)
        },
        "kd_gain": round(
            kd_accuracy - plain_accuracy,
            4
        )
    }

    return result

# Step 18 - compression_report
def compression_report(
    model,
    X_test,
    y_test,
    bits=32,
    masks=None,
    latency_iters=20
):
    # Total number of parameters.
    total_params, _ = count_params(model)

    # Dense storage assuming weights use `bits` bits and
    # non-weight parameters (biases) remain float32.
    size_mb = quantized_size_mb(model, bits)

    # Sparse storage is only applicable when pruning masks
    # are supplied.
    if masks is not None:
        sparse_mb = sparse_storage_mb(
            model,
            masks,
            value_bits=bits
        )
    else:
        sparse_mb = size_mb

    # Evaluate accuracy.
    acc = accuracy(model, X_test, y_test)

    # Measure CPU single-example latency.
    latency = measure_latency_ms(
        model,
        iters=latency_iters
    )

    return {
        "params": int(total_params),
        "sparsity": float(sparsity(model)),
        "size_mb": float(size_mb),
        "sparse_mb": float(sparse_mb),
        "accuracy": round(float(acc), 4),
        "latency_ms": round(float(latency), 3),
    }

def meets_budget(report, budget):
    size_ok = report["sparse_mb"] <= budget["max_mb"]
    latency_ok = report["latency_ms"] <= budget["max_latency_ms"]
    accuracy_ok = report["accuracy"] >= budget["min_accuracy"]

    ok = size_ok and latency_ok and accuracy_ok

    return {
        "size_ok": bool(size_ok),
        "latency_ok": bool(latency_ok),
        "accuracy_ok": bool(accuracy_ok),
        "ok": bool(ok),
    }

# Step 19 - compress_for_budget
def compress_for_budget(
    teacher,
    X_train,
    y_train,
    X_test,
    y_test,
    budget,
    student_widths=(8, 16, 32),
    prune_target=0.8,
    prune_rounds=2,
    bits=8,
    epochs=2,
    seed=0
):
    stages = []

    # ---------------------------------------------------------
    # Stage 1: Teacher (FP32, no pruning masks)
    # ---------------------------------------------------------
    teacher_report = compression_report(
        teacher,
        X_test,
        y_test,
        bits=32,
        masks=None
    )

    teacher_checks = meets_budget(
        teacher_report,
        budget
    )

    stages.append(
        ("teacher", teacher_report, teacher_checks)
    )

    if teacher_checks["ok"]:
        return {
            "stages": stages,
            "final_stage": "teacher",
            "ok": True,
            "model": teacher
        }

    # ---------------------------------------------------------
    # Stage 2: Distilled student
    # ---------------------------------------------------------
    c1, c2, hidden = student_widths

    # Use the supplied seed for deterministic student initialization.
    torch.manual_seed(seed)

    student = SmallCNN(
        c1=c1,
        c2=c2,
        hidden=hidden,
        n_classes=10
    )

    train_student(
        student,
        teacher,
        X_train,
        y_train,
        epochs=epochs,
        seed=seed
    )

    distilled_report = compression_report(
        student,
        X_test,
        y_test,
        bits=32,
        masks=None
    )

    distilled_checks = meets_budget(
        distilled_report,
        budget
    )

    stages.append(
        ("distilled", distilled_report, distilled_checks)
    )

    if distilled_checks["ok"]:
        return {
            "stages": stages,
            "final_stage": "distilled",
            "ok": True,
            "model": student
        }

    # ---------------------------------------------------------
    # Stage 3: Iterative pruning
    # ---------------------------------------------------------
    masks, _ = iterative_prune(
        student,
        X_train,
        y_train,
        target=prune_target,
        n_rounds=prune_rounds,
        epochs_per_round=1,
        seed=seed
    )

    pruned_report = compression_report(
        student,
        X_test,
        y_test,
        bits=32,
        masks=masks
    )

    pruned_checks = meets_budget(
        pruned_report,
        budget
    )

    stages.append(
        ("pruned", pruned_report, pruned_checks)
    )

    if pruned_checks["ok"]:
        return {
            "stages": stages,
            "final_stage": "pruned",
            "ok": True,
            "model": student
        }

    # ---------------------------------------------------------
    # Stage 4: Quantization of the pruned student
    # ---------------------------------------------------------
    quantized_student, _ = quantize_model(
        student,
        bits=bits,
        per_channel=True
    )

    quantized_report = compression_report(
        quantized_student,
        X_test,
        y_test,
        bits=bits,
        masks=masks
    )

    quantized_checks = meets_budget(
        quantized_report,
        budget
    )

    stages.append(
        ("quantized", quantized_report, quantized_checks)
    )

    return {
        "stages": stages,
        "final_stage": "quantized",
        "ok": bool(quantized_checks["ok"]),
        "model": quantized_student
    }

