# On-Device Model Compression Pipeline

The interview question 'a 150 MB model at 200 ms per frame has to run at 30 FPS on a phone: what do you change?' built end to end in PyTorch. Train a teacher CNN on the real Fashion-MNIST files and give it a budget (megabytes, milliseconds, minimum accuracy). Then apply the three compression levers the way practitioners do: global magnitude pruning with masked fine-tuning on a cubic sparsity schedule, symmetric and per-channel weight quantization with calibrated activation ranges and a measured signal-to-noise ratio, and knowledge distillation into a narrower student. Measure size, latency and accuracy after every stage, and finish with a pipeline that composes distill, prune and quantize until the budget is met, reporting exactly what each lever bought.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** load_fashion_mnist
- [x] **2.** SmallCNN
- [x] **3.** train_classifier
- [x] **4.** model_size_mb
- [x] **5.** measure_latency_ms
- [x] **6.** global_magnitude_prune
- [x] **7.** fine_tune_pruned
- [x] **8.** iterative_prune
- [x] **9.** sparse_storage_mb
- [x] **10.** quantize_symmetric
- [x] **11.** quantize_per_channel
- [x] **12.** quantize_model
- [x] **13.** quantization_snr_db
- [x] **14.** activation_ranges
- [x] **15.** distillation_loss
- [x] **16.** train_student
- [x] **17.** compare_students
- [x] **18.** compression_report
- [x] **19.** compress_for_budget

## Results

```
teacher: 105,866 params, 0.4235 MB fp32, 0.316 ms/frame (3164.6 FPS), accuracy 0.807

pruning to 90% sparsity: one-shot accuracy 0.586 vs iterative 0.819  (schedule [0.6333, 0.8667, 0.9])
  sparse storage 0.0639 MB vs dense 0.4235 MB; break-even sparsity 0.3333

weight quantization of the teacher:
   8-bit per-channel: 0.1062 MB, accuracy 0.809, fc1 SNR 41.18 dB
   4-bit per-channel: 0.0534 MB, accuracy 0.814, fc1 SNR 15.99 dB
   2-bit per-channel: 0.0269 MB, accuracy 0.266, fc1 SNR 1.43 dB
  measured gain per bit on fc1 (per-tensor): 5.99 dB (theory: 6.02)
  calibrated activation absmax: conv1 1.62, conv2 3.47, fc1 33.48, fc2 21.63

student (8,16,32): 26,698 params, 0.1068 MB
  distilled accuracy 0.753 vs plain 0.74 (gain +0.013); teacher 0.807
  (toy scale: the gain moves by a point or two across seeds)

budget: <= 0.03 MB, <= 33.3 ms per frame (30 FPS), accuracy >= 0.707
  (latency on this CPU is far under the 30 FPS line, so size and accuracy decide; on a phone profiler this same table is what you would show)
  teacher   0.4235 MB  0.318 ms  acc 0.807  sparsity 0.00  [size/latency/acc nYY]
  distilled 0.1068 MB  0.248 ms  acc 0.753  sparsity 0.00  [size/latency/acc nYY]
  pruned    0.0322 MB  0.251 ms  acc 0.762  sparsity 0.80  [size/latency/acc nYY]
  quantized 0.0160 MB  0.245 ms  acc 0.762  sparsity 0.80  [size/latency/acc YYY]
budget met: True at stage 'quantized'
```
