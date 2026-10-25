"""mini-Clef's synthetic tickets, through Dew.

  MODE=laya  python dew_run.py   # zero-shot Laya (released decision model) on the same eval sets as mini-Clef
  MODE=qwen  python dew_run.py   # Qwen3-0.6B + LoRA + fresh decision head, trained with DecisionObjective

Eval sets are identical to ../miniclef/train.py: seen families seed 2, held-out families seed 3.
Needs ../miniclef/data.py next to this file (cp ../miniclef/data.py .).
"""
import json, os, time
from dew.decision import Choice, Decide, Example, Noul, Score
from data import HELDOUT_FAMILIES, TRAIN_FAMILIES, dataset

MODE = os.environ.get("MODE", "laya")
N_TRAIN, N_EVAL = int(os.environ.get("N_TRAIN", 80)), int(os.environ.get("N_EVAL", 80))
OUT = os.environ.get("OUT", f"out-{MODE}"); os.makedirs(OUT, exist_ok=True)
log = open(f"{OUT}/log.txt", "a")
def say(*a):
    s = " ".join(map(str, a)); print(s, flush=True); log.write(s + "\n"); log.flush()


def to_dew(q):
    if q["type"] == "noul": return Noul(q["instructions"])
    if q["type"] == "choice": return Choice(q["instructions"], q["criteria"])
    return Score(q["instructions"], q["criteria"])


def example(rec):
    return Example(rec["state"], {k: to_dew(v) for k, v in rec["questions"].items()}, dict(rec["gold"]))


def evaluate(decide, recs, name):
    per, t0 = {}, time.time()
    for rec in recs:
        answers = decide(rec["state"], {k: to_dew(v) for k, v in rec["questions"].items()})
        for qid, gold in rec["gold"].items():
            a, q = answers[qid], rec["questions"][qid]
            if q["type"] == "noul": pred = "true" if a.noul >= 0.5 else "false"; n = 2
            elif q["type"] == "choice": pred = a.choice; n = len(q["criteria"])
            else: pred = str(max(range(len(a.probabilities)), key=lambda i: a.probabilities[i])); n = len(q["criteria"])
            v = per.setdefault(rec["families"][qid], [0, 0, 0.0]); v[0] += pred == gold; v[1] += 1; v[2] += 1 / n
    tot = sum(v[1] for v in per.values())
    out = {"set": name, "acc": round(sum(v[0] for v in per.values()) / tot, 3),
           "chance": round(sum(v[2] for v in per.values()) / tot, 3), "seconds": round(time.time() - t0),
           "families": {f: {"acc": round(v[0] / v[1], 3), "n": v[1]} for f, v in sorted(per.items())}}
    say("EVAL", json.dumps(out)); return out


seen, held = dataset(N_EVAL, 2, TRAIN_FAMILIES), dataset(N_EVAL, 3, HELDOUT_FAMILIES, k=(1, 3))
results = {"mode": MODE}

if MODE == "laya":
    decide = Decide.from_pretrained("convaiinnovations/laya")
    results["seen"], results["heldout"] = evaluate(decide, seen, "seen"), evaluate(decide, held, "heldout")
else:
    from dew import Trainer
    from dew.config import OptimConfig, TrainerConfig
    from dew.data import Memory
    from dew.decision import Brier, DecisionObjective, LogLoss, StateFirstLayout, decisions
    from dew.interop import Pretrained
    from dew.lora import LoRA
    placement = TrainerConfig().source_placement(frozen=True)
    qwen = Pretrained.load("Qwen/Qwen3-0.6B", **placement).adapt(LoRA(rank=16, modules=("q_proj", "v_proj")), key=0)
    layout = StateFirstLayout(max_len=192, head_max_len=144, option_tokens=10)  # batch 8 @ 256 OOMs on 12 GB
    objective = DecisionObjective(qwen, loss=LogLoss() + 0.5 * Brier(), layout=layout)
    table = lambda rows: Memory([{"example": example(r)} for r in rows]).pipe(decisions).read("memory")
    train = dataset(N_TRAIN, 1, TRAIN_FAMILIES)
    steps = int(os.environ.get("STEPS", 30))
    t0 = time.time()
    state = Trainer(objective, OptimConfig(learning_rate=float(os.environ.get("LR", 1e-4))), key=0).fit(
        table(train).shuffle().repeat().batch(int(os.environ.get("BATCH", 4))), steps=steps, log_every=10)
    results["train_seconds"] = round(time.time() - t0)
    decide = objective.pipeline(state)
    results["seen"], results["heldout"] = evaluate(decide, seen, "seen"), evaluate(decide, held, "heldout")

json.dump(results, open(f"{OUT}/receipt.json", "w"), indent=2)
say("DONE")
