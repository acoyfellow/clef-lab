# mini-Clef

Clef's released joint schema head on a small frozen backbone, trained on CPU, to learn the Clef recipe end to end.

- Backbone: `Qwen/Qwen3-0.6B`, frozen (Clef freezes Qwen3.8-27B; Clef-flash Qwen3.5-9B).
- Head: `JointSchemaHead` from Cloudflare's `joint_schema_model.py` (Apache-2.0, copied unchanged from
  `huggingface.co/Cloudflare/clef-flash`), at width 256 / 1 routing layer / 2 decoder layers / 4 heads (3.9M params).
  Clef-flash's head: width 1024, 2 routing, 4 layers, 16 heads, ff 4096.
- Input encoding: Clef's own `encode_record` (STATE + SCHEMA FIELDS spans).
- Loss: label-smoothed cross-entropy (0.1) + 0.5 x Brier, as described for Clef.
- Data (`data.py`): synthetic support tickets from a latent world; question order, wording, option order and option
  ids shuffled; 30% of options get random codes so the model must read descriptions.
- Not done yet vs Clef: LoRA on the backbone (Clef uses rank 256), RLCD, scale.

## Results: run1 (300 train / 80 per eval set / 10 epochs, CPU)

| Set | Untrained | Trained (epoch 10) | Chance | Brier before -> after |
|---|---|---|---|---|
| Seen families, new samples | 0.427 | **0.761** | 0.334 | 0.664 -> 0.340 |
| Held-out families (never trained) | 0.348 | **0.548** (peak 0.684 at epochs 6-7) | 0.438 | 0.565 -> 0.495 |

Per family after training:

| Family | Acc | n | |
|---|---|---|---|
| tier (choice, 4) | 1.000 | 55 | seen |
| team (choice, 5) | 0.820 | 50 | seen |
| outage (yes/no) | 0.723 | 47 | seen |
| refund (yes/no) | 0.667 | 42 | seen |
| urgency (score, 4) | 0.500 | 40 | seen |
| error_code (yes/no) | **0.796** | 49 | held out |
| tone (choice, 3) | 0.448 | 58 | held out, near chance 0.33 |
| big_amount (yes/no, "> $100") | 0.417 | 48 | held out, **below** chance 0.5 |

What this shows:
- The pipeline works and learns: seen-family accuracy more than doubles over chance, Brier halves.
- Some transfer to unseen question types (error_code), none for numeric thresholds or tone at this size.
- Held-out accuracy peaked at epoch 6-7 and then fell: the head overfits with 300 records. No early stopping was used,
  and the saved head is epoch 10. Held-out numbers were watched each epoch, so treat them as dev, not test.
- 80 eval records per set: differences under ~0.08 are noise.

Files: `results/run1/receipt.json` (all numbers), `train.log`, `joint_head.safetensors`, `joint_head_config.json`.

## Run

```sh
uv venv .venv && . .venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cpu && uv pip install transformers safetensors huggingface_hub
N_TRAIN=300 N_EVAL=80 EPOCHS=10 OUT=run1 python train.py   # ~2.4 s/record backbone pass on 4 CPU cores
```

On a Mac, switch the device to `mps` for a large speedup (not done here).

## Next

1. Run on a GPU or Apple Silicon: 5k+ records, early stopping on a separate dev set, a true test set.
2. Add LoRA on the backbone (peft), as Clef does.
3. Step up to Qwen3-1.7B / 4B; compare against real Clef-flash on the same held-out set.
4. `cog/` is an untested Cog wrapper for the Workers AI path (no Docker/GPU in the sandbox).
