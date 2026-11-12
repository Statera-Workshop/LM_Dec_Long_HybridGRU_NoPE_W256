# LM_Dec_Long_HybridGRU_NoPE_W256

HybridGRU_NoPE_W256 - Long family, decoder-only byte-level language model, standardised in the Statera Working Group Language Model library.

## Status

- Verification level: **Benchmarked**
- Failure class: none

## Provenance

- Source repository: (new) hybrid with sliding-window attention
- Upstream / license: lucidrains/x-transformers (MIT), pip 2.31.7 + torch.nn

## Differences from the source ([Unified] protocol)

- hyb_gru_nope with attention limited to the last 256 tokens

All models share one protocol: 256-byte vocabulary, dim 512, 6 layers, 8 heads x 64, Adam 2e-4 with 200 warm-up steps, gradient clip 0.5, bf16 autocast (automatic fp32 fallback). Numbers rank structures under one budget on one GPU (RTX 3070 Ti 8 GB); they do not reproduce paper results.

## Results

- Parameters: 20.74 M
- Causality check: pass (largest change of earlier outputs 0.0e+00)
- Overfit check (4 x 256 bytes): 5.696 -> 0.001 nats, pass
- LM-02 enwik8 (200M tokens): valid 1.432 / test 1.419 bits per byte
- Peak GPU memory at length 2048 (batch 1): train 823 MB, inference 272 MB
- Peak GPU memory at length 8192 (batch 1): train 9090 MB, inference 3056 MB
- LM-05 recall `lm05v3`: 4 pairs @ 512: 0.25 (keys removed 0.27); 4 pairs @ 2048: 0.02 (keys removed 0.02); 16 pairs @ 512: 0.54 (keys removed 0.06); 16 pairs @ 2048: 0.02 (keys removed 0.01)
- LM-05 recall `lm05v3long`: 4 pairs @ 512: 0.79 (keys removed 0.09); 4 pairs @ 2048: 0.01 (keys removed 0.02); 4 pairs @ 8192: 0.01 (keys removed 0.01); 16 pairs @ 512: 0.97 (keys removed 0.02); 16 pairs @ 2048: 0.01 (keys removed 0.02); 16 pairs @ 8192: 0.01 (keys removed 0.01)
- LM-06 memory update `lm06`: @512: latest 0.96, unchanged 0.14, stale 0.00; @2048: latest 0.05, unchanged 0.02, stale 0.02
- LM-06 memory update `lm06long`: @512: latest 1.00, unchanged 1.00, stale 0.00; @2048: latest 0.07, unchanged 0.02, stale 0.01; @8192: latest 0.01, unchanged 0.02, stale 0.01

## Tests

- check: data counts, causality (no look-ahead, outputs use the current byte), parameters
- scaling: peak GPU memory and time vs length 512-8192
- overfit: memorise 4 x 256 bytes below 0.5 nats
- LM-02: enwik8 next-byte prediction (bits per byte)
- LM-05: long-horizon key-value recall with key-removed control (v3: distinct queries)
- LM-06: memory update - latest value vs stale value after a key is reassigned

## Run

```bash
pip install torch==2.5.1 einops numpy
python run.py check hyb_gru_nope_w256
python run.py lm02 hyb_gru_nope_w256      # enwik8 is downloaded to data/ on first use
```
