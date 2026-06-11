# LLM Precision Benchmark: FP32 vs FP16 vs INT4

Benchmarking LLM inference across numeric precisions on a real NVIDIA T4 GPU to measure the memory and speed tradeoffs of quantization.

---

## Results

![Benchmark Chart](benchmark_results.png)

| Precision | Memory (MB) | Speed (tok/s) | vs FP32 |
|-----------|-------------|---------------|---------|
| FP32 | 477.8 | 21.06 | baseline |
| FP16 | 242.3 | 47.84 | 2× less memory, 2.3× faster |
| INT4 | 122.8 | 20.97 | 4× less memory, similar speed |

---

## Key Findings

- **FP16 is the clear production winner on T4** — half the memory, 2.3× faster, and word-for-word identical output quality vs FP32
- **INT4 saves the most memory** (4× smaller than FP32) but doesn't improve speed on T4
- **Why INT4 is slow on T4:** The T4 lacks native INT4 compute units, so bitsandbytes must decompress INT4 → FP16 at runtime, adding overhead. Newer GPUs like A100/H100 handle INT4 natively and don't have this bottleneck.
- **Output quality:** FP32 and FP16 produced identical text. INT4 produced coherent but slightly different output — expected due to precision loss.

---

## Stack

- Python, PyTorch
- HuggingFace Transformers + bitsandbytes
- matplotlib
- Hardware: NVIDIA T4 
- Model: facebook/opt-125m

---

## How to Run

1. Open in Google Colab
2. Set runtime to T4 GPU 
3. Run all cells top to bottom
