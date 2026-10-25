# clef-lab

**Learn how a decision model like Clef works by building a small one yourself, on a laptop CPU, in about an hour.**

This repo takes the open-source [Clef](https://huggingface.co/Cloudflare/clef) model apart, rebuilds its core on a
tiny backbone, trains it on synthetic data, and measures what it learned. It also includes a small real-world use of
Clef: a second opinion for regex-based security checks.

Everything here is reproducible, and every number below comes from files in [`results/`](miniclef/results/).

---

## What is a decision model?

Most language models write text. A **decision model** doesn't. You give it a situation and a few typed questions, and
it returns a probability for every allowed answer, in one fast pass:

```json
{
  "state": "Checkout has been failing for every customer for the last hour.",
  "questions": {
    "urgent":   { "type": "noul",   "instructions": "Is this urgent?" },
    "team":     { "type": "choice", "instructions": "Which team?",
                  "criteria": { "billing": "Payments and invoices", "technical": "Outages and errors" } },
    "severity": { "type": "score",  "criteria": ["None", "Minor", "Major", "Critical"] }
  }
}
```

Every answer is a probability over a fixed set of options (yes/no, named choices, or an ordered scale). There's no
output to parse, no made-up categories, and the same input gives the same answer. That makes these models useful
inside software and agent loops, wherever code needs to branch on a judgment: route a ticket, gate an action, flag a
risk, decide whether to ask a human.

These are sometimes called **System One (S1)** models: fast, intuitive, bounded. [Clef](https://huggingface.co/Cloudflare/clef)
and Clef-flash are open-weight (Apache-2.0) examples; Clairic is a sibling that extracts typed values (spans, numbers,
dates) instead of choosing between options.

## Why this repo exists

Decision models are new, and most people meet them as an API. The weights and model code for Clef are public, which
means you can see exactly how one works. But a 9B or 27B model needs a large GPU to touch.

This repo answers three questions:

1. **How is a decision model actually built?** Read [`notes/clef-recipe.md`](notes/clef-recipe.md).
2. **Can you train one yourself, small, on ordinary hardware?** Yes. See [`miniclef/`](miniclef/).
3. **What is one good for in practice?** See [`doubt/`](doubt/) for a worked example.

## How Clef works, in one picture

```
 state + questions
        |
        v
 [ frozen Qwen backbone ]   one pass over the input, no text generation
        |
        v  last hidden states
 [ joint schema head ]      each option gathers evidence from the input,
        |                   questions attend to each other, every option is scored
        v
 one probability per option, per question
```

- **Backbone:** a Qwen language model (27B for Clef, 9B for Clef-flash), frozen during training except for
  low-rank adapters (LoRA).
- **Head:** a small transformer that reads where each question and option sits in the input, and scores options
  jointly. This is where the "decision" happens.
- **Training:** synthetic records with shuffled question order, wording and schema shape; a cross-entropy loss with
  label smoothing plus a Brier loss so the probabilities are calibrated; then a reinforcement-learning stage.

Details and sources: [`notes/clef-recipe.md`](notes/clef-recipe.md).

## What we built: mini-Clef

[`miniclef/`](miniclef/) reuses Clef's **own released head and input encoding**, unchanged, on top of a backbone
about 15x smaller than Clef-flash's, and trains it on a CPU.

| | Clef-flash | mini-Clef |
|---|---|---|
| Backbone | Qwen3.5-9B (frozen + LoRA) | Qwen3-0.6B (frozen, no LoRA yet) |
| Head | width 1024, 4 layers | width 256, 2 layers (3.9M parameters) |
| Training data | large synthetic sets | 300 synthetic support tickets |
| Hardware | GPUs | 4 CPU cores |
| Loss | label-smoothed cross-entropy + Brier, then RL | label-smoothed cross-entropy + Brier |

**The data** ([`miniclef/data.py`](miniclef/data.py)) is generated from a hidden "world" (product, urgency, plan,
tone, refund request, amount, error code), turned into a support message, then asked about with randomized questions.
Option order, wording and question IDs are shuffled, and 30% of options get random codes like `Q47`, so the model has
to read the option descriptions rather than memorize labels.

**The test that matters:** three question types (tone, "amount over $100", "contains an error code") never appear in
training. Clef's main promise is answering questions it was never trained on, so that's what we measure.

### Results

| Question types | Untrained | Trained | Random chance |
|---|---|---|---|
| Seen in training (new examples) | 43% | **76%** | 33% |
| Never seen in training | 35% | **55%** (peak 68%) | 44% |

Calibration improved too: the Brier score on seen types halved, from 0.66 to 0.34.

What we learned:

- **The recipe works at small scale.** A 4M-parameter head on a frozen 0.6B model learns real decisions from 300
  examples in a few minutes of CPU training.
- **Transfer to new questions is partial.** "Does it contain an error code?" transferred well (80%). Tone and
  numeric thresholds ("over $100") did not. That matches Clef's own published scores, where the smaller Clef-flash
  is much weaker than Clef on some tasks: size matters for generalization.
- **It overfits quickly.** Accuracy on unseen types peaked around epoch 6, then fell. More data and early stopping
  are the obvious next steps.
- **Treat the numbers as a first run, not a benchmark:** 80 test records per set, synthetic data, one seed.

Per-question breakdown and how to reproduce: [`miniclef/README.md`](miniclef/README.md).

## A practical example: doubt

[`doubt/`](doubt/) shows a decision model doing a job that pattern matching does badly.

Many security tools decide whether a web server checks who's calling it by searching the code for patterns like
`Authorization` or `Bearer ${env.TOKEN}`. That's fast, but it can't tell **checking an incoming request** from
**sending an outgoing one**, which is exactly where real mistakes hide.

doubt asks Clef one question about the code ("does every request path verify the caller before touching secrets or
data?") and is only allowed to make a verdict **stricter**, never looser. On 16 planted test cases (9 open, 7 protected) across Workers,
Hono, Express, Node, Deno, Bun and FastAPI:

| | Open servers wrongly marked "protected" |
|---|---|
| Regex patterns alone | 5 of 9 |
| Regex + Clef (stricter only) | **0 of 9** |

Clef on its own got all 16 right, including traps like a token used only for an outgoing call, auth applied to only
one route, and a check that covers `GET` but not `POST`. It runs as a Cloudflare Worker behind a caller key. Details
and limits: [`doubt/RESULTS.md`](doubt/RESULTS.md).

## Try it

**Train mini-Clef** (Python 3.10+, about 1.5 GB download, about 30 minutes on 4 CPU cores):

```sh
cd miniclef
uv venv .venv && . .venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install transformers safetensors huggingface_hub
N_TRAIN=300 N_EVAL=80 EPOCHS=10 OUT=run1 python train.py
```

Most of the time goes into one pass of the backbone over every record. On Apple Silicon or a GPU it's much faster.

**Run the doubt eval** (Node 20+, a Cloudflare account with Workers AI):

```sh
cd doubt
node eval.mjs                                     # mock model: checks the wiring only
CF_ACCOUNT_ID=... CF_API_TOKEN=... node eval.mjs  # real Clef calls
```

**Use the real Clef** instead of the mini one: it's hosted on Workers AI as `@cf/cloudflare/clef` and
`@cf/cloudflare/clef-flash`, and the weights are on [Hugging Face](https://huggingface.co/Cloudflare/clef).

## Where to go next

- **Bigger and better mini-Clef:** more data, a separate test set, early stopping, LoRA on the backbone, a 1.7B or
  4B backbone, and a side-by-side against real Clef-flash.
- **Ship your own:** [`miniclef/cog/`](miniclef/cog/) is a starting point for packaging a model with
  [Cog](https://github.com/replicate/cog) (untested).
- **New kinds of decision models:** [`notes/s1-model-types.md`](notes/s1-model-types.md) lists output types nobody
  covers well yet (ranking, picking a line in the input, what changed between two states, calibrated numbers) and
  where Clef's own scores show gaps.
- **What runs before the model:** [`notes/s0-and-below.md`](notes/s0-and-below.md) covers "S0": cheap on-device
  sensing (face, hand and pose tracking in the browser, audio levels, change detection) that decides when to call a
  decision model and what to send it, with examples from [Glance](https://github.com/acoyfellow/glance) and
  [read-the-room](https://github.com/acoyfellow/read-the-room).
- **Side note:** [`notes/handwired-in-workers.md`](notes/handwired-in-workers.md) covers running
  [Handwired](https://github.com/chenglou/handwired) programs inside a Cloudflare Worker.

## Repo layout

```
miniclef/
  data.py                 synthetic records, held-out question types
  train.py                cache backbone states, train the head, evaluate, export
  joint_schema_model.py   Clef's model code, unchanged (Apache-2.0)
  cog/                    Cog packaging sketch (untested)
  results/run1/           receipt.json, train.log, trained head + config
doubt/
  src/                    Worker + review logic
  corpus/cases.mjs        16 planted server examples
  eval.mjs, RESULTS.md
notes/
  clef-recipe.md          how Clef is built, trained and served
  s1-model-types.md       ideas for new decision models
  s0-and-below.md         on-device sensing before S1 and LLMs
dew/                      same data through the Dew framework (Laya zero-shot, Qwen + LoRA)
dew/                      same data through the Dew framework (Laya zero-shot, Qwen + LoRA)
  handwired-in-workers.md
```

## Credits and license

- Clef, Clef-flash and `joint_schema_model.py` are by Cloudflare, released under Apache-2.0 on
  [Hugging Face](https://huggingface.co/Cloudflare/clef). The file is included here unchanged.
- The mini-Clef backbone is [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) (Apache-2.0).
- The question format follows the Jev / SystemOne decision-model API that Clef is compatible with.

This repo is licensed under Apache-2.0. See [`LICENSE`](LICENSE).
