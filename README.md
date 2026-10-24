# clef-lab

Exploring Clef-style decision models: how they're built, a small reproduction, and a Clef-backed check.

| Path | What | Result |
|---|---|---|
| [`miniclef/`](miniclef/) | Clef's joint schema head on frozen Qwen3-0.6B, synthetic data, CPU training, Cog wrapper | seen question types 0.43 -> 0.76 (chance 0.33); unseen types 0.35 -> 0.55 (chance 0.44) |
| [`doubt/`](doubt/) | escalation-only Clef second opinion for regex auth detectors (Worker) | 16 cases: regex wrongly passes 5 open handlers -> 0 with Clef |
| [`notes/clef-recipe.md`](notes/clef-recipe.md) | architecture, training, serving | from public sources |
| [`notes/s1-model-types.md`](notes/s1-model-types.md) | other S1 model types worth building | ideas |
| [`notes/handwired-in-workers.md`](notes/handwired-in-workers.md) | Handwired in workerd | tested |

`miniclef/joint_schema_model.py` is Cloudflare's file from huggingface.co/Cloudflare/clef-flash (Apache-2.0), unchanged.
