# mini-Clef's data, through Dew

[Dew](https://dewml.dev) is a JAX/Flax training framework with a decision-model module (`dew.decision`): Jev-style
`Noul` / `Choice` / `Score` questions, Laya and Clef-style joint-head checkpoints loaded natively, a
`DecisionObjective` that puts a decision head on any backbone (frozen or LoRA), proper scoring-rule losses,
calibration and abstention, and GGUF export for llama.cpp.

This folder runs the same synthetic tickets and the **same eval sets** as [`../miniclef`](../miniclef) through Dew,
on CPU.

## Results

| Model | Training | Seen question types | Never-seen question types |
|---|---|---|---|
| Laya (ModernBERT-large), zero-shot | none | **78.6%** | **84.5%** |
| Qwen3-0.6B + LoRA, Dew `DecisionObjective` | 30 steps x 4 rows, 7.7 min | 63.7% | 63.2% |
| mini-Clef (frozen Qwen3-0.6B + Clef head) | 300 records x 10 epochs | 76.1% | 54.8% |
| random chance | | 33.4% | 43.8% |

Per question type: [`results.json`](results.json).

What this says:

- **A released decision model beats everything we trained, without seeing our data.** Laya got 100% on "amount
  over $100" and 98% on "contains an error code", two types mini-Clef never learned. Generalizing to new questions
  is what pretraining on many decision tasks buys, and a few hundred synthetic tickets can't replace it.
- **Dew's LoRA run generalizes better than mini-Clef after far less training** (63% vs 55% on unseen types, after
  about 120 question rows). Adapting the backbone, not just a head on top of a frozen one, seems to matter.
- **Urgency (a 4-level score) is hard for every model here.** It's also the most subjective label in the generator.
- Small sets (80 records each), one seed, CPU-sized runs: treat as a first look.

## Run

```sh
uv venv -p 3.12 .venv && . .venv/bin/activate
uv pip install "dewml @ git+https://github.com/AshishKumar4/dew" --extra-index-url https://pypi.dewml.dev/simple
cp ../miniclef/data.py .
MODE=laya python dew_run.py      # ~4 min on 3 CPU cores
MODE=qwen python dew_run.py      # ~8 min train + ~5 min eval, ~8 GB RAM
```

Batch 8 at 256 tokens ran a 12 GB machine out of memory. On a GPU, use more data and steps, a separate dev set,
and `decide.calibrated(...)`.

## Fine-tuning Laya on a 12 GB CPU machine

Laya is the strongest model here, so the next step is fine-tuning it. What fits in 12 GB of RAM:

| Setup | Trainable | Peak memory | Result |
|---|---|---|---|
| full fine-tune, float32 | 421M | > 12.3 GB | killed (out of memory) |
| full fine-tune, `dtype="bfloat16"`, batch 1 | 421M | ~12.3 GB | thrashes; parameters stay float32, only compute is bf16 |
| head + every norm layer | 26.3M | > 12 GB | killed: norms sit in every block, so backprop still runs through the whole backbone |
| **head + final norm only** | **26.2M** | **6.7 GB** | **fits**: ~22 s/step at batch 4; train loss 0.87 -> 0.43 by step 40 (no eval yet) |

```python
from dew.objectives.base import freeze
head_and_final_norm = lambda path: len(path) > 2 and (
    path[1] == "head" or (path[1] == "backbone" and path[2] == "norm"))
objective = DecisionObjective(laya, loss=LogLoss() + 0.5 * Brier(),
                              variables=freeze(laya.variables, head_and_final_norm))
```

**Gotcha:** freezing the *entire* backbone (`path[1] == "head"` only) fails with `KeyError: 'params'` in
`thaw()`, called from `DecisionModel._read`: the backbone subtree then has a `frozen` collection but no `params`.
Keeping any one backbone leaf trainable (here the final norm) avoids it.

## Next

- Fine-tune Laya itself on the tickets (start from the strongest model, not a fresh head).
- Train longer, then score Laya, Qwen+LoRA and Clef-flash's joint-head release side by side in Dew's harness.
- Export with `save_pretrained` and serve the GGUF with `llama-server` at `/v1/systemone`.
