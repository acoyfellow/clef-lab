"""Fine-tune Laya on mini-Clef's synthetic tickets; eval on the same sets as dew_run.py.

Trains Laya's decision head plus the backbone's final norm (backbone frozen otherwise).
Freezing the whole backbone hits https://github.com/AshishKumar4/dew/issues/50.

  cp ../miniclef/data.py .
  python laya_ft.py                      # head + final norm, ~7 GB RAM
  FULL=1 python laya_ft.py               # full fine-tune (needs >12 GB RAM)
"""
import json, os, time
from dew import Trainer
from dew.config import OptimConfig
from dew.data import Memory
from dew.decision import Brier, Choice, Decide, DecisionObjective, Example, LogLoss, Noul, Score, decisions
from dew.objectives.base import freeze
from data import HELDOUT_FAMILIES, TRAIN_FAMILIES, dataset

N_TRAIN = int(os.environ.get("N_TRAIN", 300))
STEPS = int(os.environ.get("STEPS", 150))
BATCH = int(os.environ.get("BATCH", 8))
FULL = os.environ.get("FULL") == "1"
OUT = os.environ.get("OUT", "out-laya-full" if FULL else "out-laya-head"); os.makedirs(OUT, exist_ok=True)
log = open(f"{OUT}/log.txt", "a")
def say(*a):
    s = " ".join(map(str, a)); print(s, flush=True); log.write(s + "\n"); log.flush()

def to_dew(q):
    if q["type"] == "noul": return Noul(q["instructions"])
    if q["type"] == "choice": return Choice(q["instructions"], q["criteria"])
    return Score(q["instructions"], q["criteria"])

def example(r):
    return Example(r["state"], {k: to_dew(v) for k, v in r["questions"].items()}, dict(r["gold"]))

def evaluate(decide, recs, name):
    per = {}
    for rec in recs:
        ans = decide(rec["state"], {k: to_dew(v) for k, v in rec["questions"].items()})
        for qid, gold in rec["gold"].items():
            a, q = ans[qid], rec["questions"][qid]
            if q["type"] == "noul": pred = "true" if a.noul >= 0.5 else "false"
            elif q["type"] == "choice": pred = a.choice
            else: pred = str(max(range(len(a.probabilities)), key=lambda i: a.probabilities[i]))
            v = per.setdefault(rec["families"][qid], [0, 0]); v[0] += pred == gold; v[1] += 1
    tot = sum(v[1] for v in per.values())
    out = {"set": name, "acc": round(sum(v[0] for v in per.values()) / tot, 3),
           "families": {f: round(v[0] / v[1], 3) for f, v in sorted(per.items())}}
    say("EVAL", json.dumps(out)); return out

seen, held = dataset(80, 2, TRAIN_FAMILIES), dataset(80, 3, HELDOUT_FAMILIES, k=(1, 3))
laya = Decide.from_pretrained("convaiinnovations/laya", dtype="bfloat16")
head_and_final_norm = lambda path: len(path) > 2 and (path[1] == "head" or (path[1] == "backbone" and path[2] == "norm"))
variables = laya.variables if FULL else freeze(laya.variables, head_and_final_norm)
objective = DecisionObjective(laya, loss=LogLoss() + 0.5 * Brier(), variables=variables)
table = lambda rows: Memory([{"example": example(r)} for r in rows]).pipe(decisions).read("memory")

t0 = time.time()
state = Trainer(objective, OptimConfig(learning_rate=float(os.environ.get("LR", 2.5e-5 if FULL else 3e-4))), key=0).fit(
    table(dataset(N_TRAIN, 1, TRAIN_FAMILIES)).shuffle().repeat().batch(BATCH), steps=STEPS, log_every=10)
train_s = round(time.time() - t0); say("TRAIN_SECONDS", train_s)
decide = objective.pipeline(state)
result = {"mode": "full" if FULL else "head+final-norm", "n_train": N_TRAIN, "steps": STEPS, "batch": BATCH,
          "train_seconds": train_s, "seen": evaluate(decide, seen, "seen"), "heldout": evaluate(decide, held, "heldout")}
json.dump(result, open(f"{OUT}/receipt.json", "w"), indent=2); say("DONE")
