"""mini-Clef: Clef's released joint schema head on a small frozen backbone, trained on CPU.
1) cache frozen-backbone hidden states  2) train JointSchemaHead with label-smoothed CE + Brier
3) eval on seen families (new samples) and fully held-out families, vs chance and vs the untrained head."""
import json, os, random, sys, time
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import joint_schema_model as jsm
from data import TRAIN_FAMILIES, HELDOUT_FAMILIES, dataset

torch.manual_seed(0); random.seed(0); torch.set_num_threads(4)
BACKBONE = os.environ.get("BACKBONE", "Qwen/Qwen3-0.6B")
N_TRAIN, N_EVAL = int(os.environ.get("N_TRAIN", 300)), int(os.environ.get("N_EVAL", 80))
OUT = os.environ.get("OUT", "out"); os.makedirs(OUT, exist_ok=True)
log = open(f"{OUT}/train.log", "a")
def say(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); log.write(s + "\n"); log.flush()

tok = AutoTokenizer.from_pretrained(BACKBONE)
lm = AutoModelForCausalLM.from_pretrained(BACKBONE, dtype=torch.float32).eval()
for p in lm.parameters(): p.requires_grad_(False)
emb_w = lm.get_output_embeddings().weight
H = lm.config.hidden_size

@torch.no_grad()
def cache(recs, tag):
    path = f"{OUT}/cache-{tag}.pt"
    if os.path.exists(path): return torch.load(path, weights_only=False)
    items, t0 = [], time.time()
    for i, rec in enumerate(recs):
        e = jsm.encode_record(tok, {"state": rec["state"], "questions": rec["questions"]}, max_length=1024)
        ids = torch.tensor([e.input_ids])
        hs = lm.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).last_hidden_state[0]
        items.append((e, ids[0], hs.half(), rec))
        if i % 25 == 0: say(f"cache {tag} {i}/{len(recs)} {time.time()-t0:.0f}s")
    torch.save(items, path); return items

tr = cache(dataset(N_TRAIN, 1, TRAIN_FAMILIES), "train")
se = cache(dataset(N_EVAL, 2, TRAIN_FAMILIES), "seen")
he = cache(dataset(N_EVAL, 3, HELDOUT_FAMILIES, k=(1, 3)), "heldout")

cfg = dict(hidden_size=H, width=256, routing_layers=1, layers=2, heads=4, feedforward=512)
head = jsm.JointSchemaHead(**cfg, dropout=0.1)

def logits_for(item):
    e, ids, hs, _ = item
    return head(hs.float().unsqueeze(0), ids.unsqueeze(0), torch.ones(1, len(ids), dtype=torch.long), [e], emb_w)[0]

def gold_index(e, rec, qi):
    return e.questions[qi].option_ids.index(rec["gold"][e.questions[qi].question_id])

@torch.no_grad()
def evaluate(items, name):
    head.eval(); per = {}; brier = []
    for item in items:
        e, _, _, rec = item
        for qi, lg in enumerate(logits_for(item)):
            p = lg.float().softmax(-1); g = gold_index(e, rec, qi)
            fam = rec["families"][e.questions[qi].question_id]
            v = per.setdefault(fam, [0, 0, 0.0]); v[0] += int(p.argmax().item() == g); v[1] += 1; v[2] += 1.0 / len(p)
            brier.append(((p - F.one_hot(torch.tensor(g), len(p)).float()) ** 2).sum().item())
    tot = sum(v[1] for v in per.values())
    return {"set": name, "acc": round(sum(v[0] for v in per.values()) / tot, 3), "chance": round(sum(v[2] for v in per.values()) / tot, 3),
            "brier": round(sum(brier) / len(brier), 3),
            "families": {f: {"acc": round(v[0] / v[1], 3), "chance": round(v[2] / v[1], 3), "n": v[1]} for f, v in sorted(per.items())}}

before = [evaluate(se, "seen"), evaluate(he, "heldout")]
say("UNTRAINED", json.dumps(before))
opt = torch.optim.AdamW(head.parameters(), lr=float(os.environ.get("LR", 3e-4)), weight_decay=0.01)
EPOCHS = int(os.environ.get("EPOCHS", 10))
for ep in range(EPOCHS):
    head.train(); random.shuffle(tr); tot, t0 = 0.0, time.time()
    for item in tr:
        e, _, _, rec = item; loss = 0.0
        for qi, lg in enumerate(logits_for(item)):
            t = torch.tensor(gold_index(e, rec, qi))
            ce = F.cross_entropy(lg.float().unsqueeze(0), t.unsqueeze(0), label_smoothing=0.1)
            p = lg.float().softmax(-1)
            loss = loss + ce + 0.5 * ((p - F.one_hot(t, len(p)).float()) ** 2).sum()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0); opt.step(); tot += loss.item()
    say(f"epoch {ep+1}/{EPOCHS} loss {tot/len(tr):.4f} {time.time()-t0:.0f}s | seen {evaluate(se,'seen')['acc']} | heldout {evaluate(he,'heldout')['acc']}")

after = [evaluate(se, "seen"), evaluate(he, "heldout")]
say("TRAINED", json.dumps(after))
from safetensors.torch import save_file
save_file({k: v.contiguous() for k, v in head.state_dict().items()}, f"{OUT}/joint_head.safetensors")
json.dump(cfg, open(f"{OUT}/joint_head_config.json", "w"), indent=2)
json.dump({"backbone": BACKBONE, "n_train": N_TRAIN, "n_eval": N_EVAL, "epochs": EPOCHS, "before": before, "after": after,
           "head_params": sum(p.numel() for p in head.parameters())}, open(f"{OUT}/receipt.json", "w"), indent=2)
say("DONE", f"head params {sum(p.numel() for p in head.parameters()):,}")
