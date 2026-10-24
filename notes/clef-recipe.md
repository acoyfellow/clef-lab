# Clef recipe (field notes)

Sources: Hugging Face `Cloudflare/clef`, `Cloudflare/clef-flash` (model cards, `joint_schema_model.py`, Apache-2.0)
and the public launch post "Introducing Clef" (blog.cloudflare.com/clef-decision-models).

## Shape

- Backbone: Qwen, frozen, one prefill pass, no generation. Clef: Qwen3.8-27B. Clef-flash: Qwen3.5-9B.
- Input: `STATE:` + state (text/JSON, images, video) + `SCHEMA FIELDS:` (ID, TYPE, INSTRUCTION, ALLOWED OPTIONS).
  The encoder records token spans for every question and option.
- Joint schema head on the last hidden states:
  - option queries = option context + option token embeddings + question vector
  - evidence routing: options cross-attend to the input
  - fields = question + routed option summary + global vector + type embedding, then decoder layers (fields attend
    to each other and the input)
  - score = lexical prior + gated (cosine + residual MLP); softmax per question
- Question types: `noul` (yes/no), `choice` (named options), `score` (ordered options).
- Clef-flash head: width 1024, 2 routing layers, 4 decoder layers, 16 heads, ff 4096.

## Training (as published)

- Head trained jointly with rank-256 LoRA on the frozen backbone.
- Loss: label-smoothed cross-entropy + Brier (calibration).
- Synthetic data permuting field order, prompts and schema structure.
- RLCD: RL for calibrated decisions (partial credit for adjacent ordinal choices, reward for fully correct records,
  reference penalty).

## Serving

- transformers: `load_release_model`, `systemone()` (Jev/SystemOne-compatible request/response).
- SGLang: `/v1/systemone`.
- Workers AI: `@cf/cloudflare/clef`, `@cf/cloudflare/clef-flash`; billed on input tokens only; 64k context;
  images as 32x32 patches, max 1,024 tokens each.
- Custom models on Workers AI are packaged with Cog (`cog.yaml` + `predict.py`). See `miniclef/cog/`.

## Reproducing at small scale

`miniclef/`: same head and encoding on frozen Qwen3-0.6B, CPU. LoRA, RLCD and scale are the next steps.
