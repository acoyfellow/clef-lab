# Systems of models: a field guide

Most useful AI software is no longer one model. It's several, each doing the job it's cheapest and most reliable
at, with ordinary code deciding who goes next. This guide maps the idea: where the language comes from, who is
building what, the patterns that keep showing up, and where it seems to be heading.

Snapshot: late October 2026. Repo star counts were read from GitHub on 2026-10-25; they move fast. Dates and
citations for older work are from the literature as commonly cited; check the originals before relying on them.

---

## 1. The idea in one picture

```
 signal ──► S0  sense locally     did anything change?            free, per frame / event
            │
            ▼
            S1  decide            typed answer + probabilities    ~10-500 ms, fractions of a cent
            │      ├─ confident ──► code acts
            │      └─ unsure ─────┐
            ▼                     ▼
            S2  reason / write    open-ended LLM work             seconds, cents to dollars
            │                     │
            ▼                     ▼
            code checks ◄─────────┘          human when nothing above is sure
```

Each layer is roughly 10-1000x cheaper than the one above it and understands less. The engineering job is to push
every decision as low as it can go **and still be right**, and to know when it isn't.

## 2. Where the language comes from

| Term | Origin | What it means here |
|---|---|---|
| **System 1 / System 2** | Dual-process psychology (Stanovich and West, ~2000), popularized by Kahneman's *Thinking, Fast and Slow* (2011) | fast, intuitive, bounded judgment vs slow, deliberate reasoning |
| **"System 2" deep learning** | Yoshua Bengio's NeurIPS 2019 keynote | ML research framing: today's nets are System 1; reasoning is the frontier |
| **System One models** | TypeSafe AI's Jev launch, Sep 15 2026 | a product category: state + typed questions in, calibrated typed decisions out, no text |
| **Decision model** | Cloudflare's Clef launch (Oct 1 2026) and the open-source ecosystem | same category, the more neutral name |
| **Compound AI systems** | Berkeley AI Research blog, Zaharia et al., Feb 2024 | state-of-the-art results increasingly come from systems of components, not single models |
| **Software 2.0** | Karpathy, 2017 | behaviour specified by data and weights instead of code; S1 models make that callable like a function |

The older computer-science roots are deeper than the branding:

| Idea | Classic work | Lives on as |
|---|---|---|
| Cascades | Viola-Jones face detector (2001): cheap stages reject most windows | S0 gates, LLM cascades |
| Mixture of experts | Jacobs, Jordan, Nowlan and Hinton (1991) | routers, MoE layers |
| Reject option / abstention | Chow (1970); selective classification (El-Yaniv and Wiener 2010; Geifman and El-Yaniv 2017) | "answer only when sure" |
| Learning to defer | Madras et al. (2018); Mozannar and Sontag (2020) | escalate to a bigger model or a person |
| Calibration | Brier score (1950); Platt scaling (1999); temperature scaling (Guo et al. 2017) | probabilities you can threshold |
| Conformal prediction | Vovk, Gammerman, Shafer (2005) | guaranteed error rates on accepted answers |
| Stacking / ensembles | Wolpert (1992); boosting | combining models, disagreement as signal |
| Early exit | BranchyNet (2016); LayerSkip (2024) | the same idea inside one network |
| Distillation | Hinton, Vinyals, Dean (2015) | big model teaches a small one |
| Zero-shot classification via NLI | Yin et al. (2019) | the ancestor of "write the options at call time" |
| ML technical debt | Sculley et al., "Hidden Technical Debt in ML Systems" (2015) | why the glue code matters more than the model |

**The short version:** cascades, abstention and calibration are 25-55 years old. What's new in 2026 is a single
general model that takes *any* schema at call time, returns calibrated probabilities in one pass, and is cheap
enough to call on every event.

## 3. Who is building what

### Decision models (S1)

| Project | What | Notes |
|---|---|---|
| **Jev** (TypeSafe AI) | the hosted model that named the category | `noul` / `choice` / `score` question types; `/v1/systemone` wire format; RLCD training |
| **Clef, Clef-flash, Clef-omni** (Cloudflare) | open-weight (Apache-2.0) Qwen backbones + joint schema head | on Workers AI and [Hugging Face](https://huggingface.co/Cloudflare/clef); see [`clef-recipe.md`](clef-recipe.md) |
| **Clairic** (Cloudflare) | extraction sibling: spans, numbers, dates, ranges, "absent" | pairs with Clef: Clairic reads, Clef decides |
| **Laya** ([convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya)) | open decision model on ModernBERT-large | very fast (single-digit ms), weaker on hard tasks |
| **Kev**, **von**, community "Jev-style" checkpoints | open alternatives and fine-tunes | quality varies; benchmark on your task |
| **GLiNER / GLiNER2** ([urchade](https://github.com/urchade/GLiNER), [fastino-ai](https://github.com/fastino-ai/GLiNER2)) | schema-at-call-time entity and information extraction | pre-dates the S1 branding; same philosophy for extraction |
| **SetFit** ([huggingface/setfit](https://github.com/huggingface/setfit)) | few-shot classifiers from sentence embeddings | the "train a tiny one" end of the spectrum |

### Tooling for training and serving them

| Project | What |
|---|---|
| **[Dew](https://dewml.dev)** ([AshishKumar4/dew](https://github.com/AshishKumar4/dew)) | JAX/Flax training library with a full `dew.decision` module: Jev question types, loads Laya and Clef-style joint-head releases, fine-tunes a decision head over any backbone (LoRA or frozen), proper scoring-rule losses (log, Brier, spherical, ranked probability), per-question calibration, abstention thresholds, risk-coverage metrics, a `systemone()` endpoint, and export that llama.cpp can serve |
| **llama.cpp** | serves decision models natively at `POST /v1/systemone` (per Dew's docs) |
| **[ollaya](https://github.com/ollaya-dev/ollaya)** | "Ollama for decision models": pull and serve many families behind the Jev API |
| **[vllm-jev](https://github.com/mode-io/vllm-jev)**, **[openjev-sglang](https://github.com/ekzhang/openjev-sglang)** | serving on the big inference engines (prefill-only) |
| **[JevBench](https://github.com/fstandhartinger/jevbench)**, Decision Index | cross-model benchmarks for typed decisions |

### Routers and cascades (choosing which model runs)

| Project | What |
|---|---|
| **[RouteLLM](https://github.com/lm-sys/RouteLLM)** (LMSYS) | learned routers between a strong and a weak LLM |
| **[FrugalGPT](https://github.com/stanford-futuredata/FrugalGPT)** (Stanford) | the paper that made LLM cascades a cost strategy |
| **[vLLM Semantic Router](https://github.com/vllm-project/semantic-router)** | "an open, programmable decision layer for models and compute" |
| **[aurelio-labs/semantic-router](https://github.com/aurelio-labs/semantic-router)** | embedding-based fast routing |
| **[LLMRouter](https://github.com/ulab-uiuc/LLMRouter)**, **[plano](https://github.com/katanemo/plano)** | routing libraries and proxies |
| **[Agreement-Based Cascading](https://github.com/stevenkolawole/Agreement-Based-Cascading)** | training-free cascades that defer when an ensemble disagrees |

### Programming the whole system

| Project | What |
|---|---|
| **[DSPy](https://github.com/stanfordnlp/dspy)** (Stanford) | "programming, not prompting": optimize pipelines of model calls end to end |
| **[Optimas](https://github.com/snap-stanford/optimas)** (Stanford) | optimizing compound AI systems with local rewards per component |
| **[awesome-compound-ai-systems](https://github.com/compound-ai-systems/awesome-compound-ai-systems)** | reading list |

### What people are using S1 for (from the community directory)

[awesome-typesafe-jev](https://github.com/AbdelStark/awesome-typesafe-jev) lists well over a hundred projects. The
recurring jobs:

- **Context management for coding agents:** prune long tool output, choose what survives compaction, pick which code
  paragraphs an agent needs.
- **Gates before actions:** deny or ask on risky tool calls; turn rules stated in chat into checked policy.
- **Routing:** pick the model and reasoning effort for a task.
- **Grep by meaning:** stream log lines and keep the ones a typed question flags.
- **Wake/skip for sleeping agents:** decide whether an event is worth resuming an expensive LLM.
- **Browser agents:** S1 picks the next click from observed controls; the LLM plans; code checks.
- **Reranking, data labeling, generative UI** (pick a component, fill its props), games and robotics.

## 4. Patterns that keep working

1. **Read, check, decide.** An extractor pulls values; code checks anything with a fixed format; the decision model
   makes the judgment call. (Clairic, code, Clef.)
2. **Threshold, then escalate.** Act when confident; otherwise go one layer up, and log what each layer saved.
3. **Only stricter.** A model may raise a finding over a cheap rule but never lower it, until you've measured it
   enough to trust it both ways. (See [`doubt/`](../doubt/).)
4. **Disagreement is a signal.** Two cheap judges that disagree are the cheapest error detector you have: S0 vs S1,
   rule vs model, two models.
5. **Calibrate on your data.** Raw probabilities are rarely calibrated. Fit a temperature per question type on
   held-out examples, and pick thresholds by the error you can accept, not by eye.
6. **Freeze the rule before measuring.** Write the success criterion and the test set down before the run.
7. **Ask fewer, better questions.** Clear, mutually exclusive options beat clever prompts.
8. **Keep the LLM for what only it can do,** then check what it writes with S1 and code.

## 5. What independent tests have found (and why to test your own)

From community studies linked in awesome-typesafe-jev, each specific to its dataset and model run:

- Given an "unknown" option, a decision model may choose it far more often than a human would; removing it changes
  accuracy a lot. Design abstention deliberately.
- A cascade into an LLM improved accuracy on one benchmark and only raised cost on another.
- Conformal-style thresholds can certify a bounded misroute rate on in-scope data, but out-of-scope traffic breaks
  the guarantee unless it's measured separately.
- Sorting by probability can pass on one corpus and fail on another; many items tie near 1.0.
- On a 111-case agent-action gate, a decision model and a frontier LLM performed similarly, each with an unsafe
  allow: the policy mapping was harder than the model.

Clef's own published scores show the same shape: excellent at routing, intent and tool selection; weak at
multi-label (SATA), relations (ACOS), tracking state across turns (SGD) and hard reasoning (GPQA). Our
[mini-Clef](../miniclef/) run shows it at toy scale: some unseen question types transfer, others don't.

## 6. Where this is heading

- **More output types.** Today: pick an option, read a value. Coming: subsets, rankings, pairwise preferences with a
  criterion written per call, "which line", "what changed", calibrated quantities. See [`s1-model-types.md`](s1-model-types.md).
- **A real S0 layer.** On-device sensing (MediaPipe, voice activity detection, change detection) decides when to call
  S1 at all. See [`s0-and-below.md`](s0-and-below.md).
- **Tiny per-task students.** A large S1 labels real traffic; a small model learns the one workflow, runs locally or
  at the edge, and escalates what it doesn't know. Training libraries (Dew) and RL fine-tuning services are already
  shaped for this.
- **One wire format.** The Jev `/v1/systemone` shape is becoming the "OpenAI chat completions" of decisions: hosted
  APIs, llama.cpp, vLLM and SGLang servers all speak it, so models become swappable.
- **Exact programs around learned ones.** Bounded typed outputs are exactly what exact, inspectable programs (e.g.
  [Handwired](https://github.com/chenglou/handwired)) need as inputs: learned perception, hand-written policy.
- **Systems optimized end to end.** DSPy- and Optimas-style optimization over whole pipelines, with S1 calls as cheap
  differentiable-ish steps and calibration as a first-class objective.

## 7. Get started in an afternoon

1. **Call one.** Clef on Workers AI, Jev's hosted API, or Laya locally through Dew or llama.cpp. Ask three questions
   (`noul`, `choice`, `score`) about one real input from your work.
2. **Find one `if` in your code that a human currently decides,** or that a regex gets wrong. Write 30 labelled
   examples *before* trying a model.
3. **Measure:** accuracy, calibration (are 90% answers right 90% of the time?), and cost per call against what you do
   today.
4. **Add a threshold and an escalation path,** and log both layers' answers.
5. **When it's worth it, train a small one:** [`miniclef/`](../miniclef/) shows the recipe on a laptop; Dew does it
   properly on GPUs or TPUs.

## Reading list

- Kahneman, *Thinking, Fast and Slow* (2011)
- Zaharia et al., "The Shift from Models to Compound AI Systems", BAIR blog (2024)
- Chen, Zaharia, Zou, "FrugalGPT" (2023); Ong et al., "RouteLLM" (2024)
- Guo et al., "On Calibration of Modern Neural Networks" (2017)
- Geifman and El-Yaniv, "Selective Classification for Deep Neural Networks" (2017)
- Mozannar and Sontag, "Consistent Estimators for Learning to Defer to an Expert" (2020)
- Viola and Jones, "Rapid Object Detection using a Boosted Cascade of Simple Features" (2001)
- Sculley et al., "Hidden Technical Debt in Machine Learning Systems" (2015)
- Khattab et al., "DSPy" (2023)
- TypeSafe, "Introducing System One Models & Jev" (2026); Cloudflare, "Introducing Clef" (2026)
- [Dew decision-model guide](https://dewml.dev/guides/decision-models/)
