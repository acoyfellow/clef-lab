# Findings, gaps and predictions

What this repo's experiments showed, what the ecosystem is missing, and where we think systems of models are
heading. Sections 1-2 are measured or observed; sections 3-4 are opinion. Background and sources:
[`systems-of-models.md`](systems-of-models.md).

## 1. What we measured

**A general decision model beat everything we trained, without seeing our data.** On the same synthetic support
tickets and the same 80-record eval sets ([`../dew/`](../dew/), [`../miniclef/`](../miniclef/)):

| Model | Training | Seen question types | Never-seen question types |
|---|---|---|---|
| Laya (ModernBERT-large), zero-shot | none | **78.6%** | **84.5%** |
| mini-Clef (frozen Qwen3-0.6B + Clef's head) | 300 records x 10 epochs | 76.1% | 54.8% |
| Qwen3-0.6B + LoRA via Dew | ~120 question rows, 7.7 min | 63.7% | 63.2% |
| random chance | | 33.4% | 43.8% |

- Generalizing to *new questions* is what pretraining on many decision tasks buys. A few hundred task examples
  don't replace it; they mostly teach the questions you trained on.
- Adapting the backbone (LoRA) generalized better than a head on a frozen backbone, after far less training.
- The most subjective label (a 4-level urgency score) was hard for every model.
- **Practical rule: start with a general decision model and train only when it fails on your data.**

**A decision model is a good second opinion for brittle rules.** In [`../doubt/`](../doubt/), a typical regex auth
detector wrongly passed 5 of 9 open servers (it can't tell an incoming check from an outgoing call). Letting Clef only
make verdicts *stricter* brought that to 0 of 9, with no new false alarms on 16 cases. Clef alone got 16 of 16.

All of these are small, synthetic or self-written test sets with one seed: first looks, not benchmarks.

## 2. Patterns already working in public projects

- **Read, check, decide.** [read-the-room](https://github.com/acoyfellow/read-the-room) states it as "Clairic reads,
  code checks, Clef decides", with an LLM only where nothing else can do the job and its output checked.
- **Disagreement logs.** read-the-room records an in-browser pose count next to Clairic's count of the same frame,
  and whether they agreed: two cheap judges, with disagreement as the error signal.
- **S0 in the page.** [Glance](https://github.com/acoyfellow/glance) runs face and hand tracking in the browser on
  every new frame and never calls a model for it. See [`s0-and-below.md`](s0-and-below.md).
- **Stricter-only gates.** A model may raise a cheap rule's verdict, never lower it, until it's measured both ways.

## 3. Gaps nobody seems to own yet

1. **A benchmark for the whole ladder.** Benchmarks score one model. Nobody scores a full S0 → S1 → S2 → human system
   on cost, error and latency together, which is what you actually ship.
2. **A standard way for S0 to hand hints to S1.** Local signals (`{"faces": 2, "still_ms": 1800}`) go into the state
   ad hoc. A shared shape would make sensing layers and decision models swappable.
3. **A standard decision log.** Every layer's answer, its confidence and the final outcome, in one record. That turns
   disagreements into training data automatically.
4. **Native ranking and pairwise-preference decision models,** with the criterion written per call. Today people fake
   ranking with many yes/no questions, and independent tests show it breaks on real data.

## 4. Predictions

1. **Decision models become a standard serving tier.** The Jev `/v1/systemone` wire format is already spoken by
   hosted APIs, llama.cpp, vLLM and SGLang servers. Expect every inference provider to offer prefill-only, output-free
   pricing.
2. **The value moves from the model to the question.** Writing good question sets (mutually exclusive options,
   deliberate abstention) becomes a skill, then a library, then something optimized automatically.
3. **Per-task students.** A large decision model labels real traffic, a tiny model learns one workflow and runs close
   to the data, and the large one only sees the hard cases. Section 1 says when this pays: only after the general
   model fails.
4. **Calibration becomes a contract.** "At most 2% silent error on accepted answers", backed by conformal methods, is
   a reliability guarantee customers can buy, not just a model feature.
5. **S0 moves into hardware.** Wake words, motion triggers and presence sensors emitting typed events in the same
   shape decision models read.
6. **Learned perception, exact policy.** Bounded typed outputs feed exact, inspectable programs (e.g.
   [Handwired](https://github.com/chenglou/handwired)): the model sees, hand-written logic decides what to do.
