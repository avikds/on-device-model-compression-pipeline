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

---

Built on Deep-ML.
