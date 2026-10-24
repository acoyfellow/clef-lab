# Cog predictor for a Clef-shaped decision model on Workers AI.
# Weights load in setup(), inference under torch.no_grad(), an input_tokens metric is recorded for billing.
# Untested: not yet built with Docker or run on a GPU.
import json, os, sys
import torch
from cog import BasePredictor, Input
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import load_file


class Predictor(BasePredictor):
    def setup(self):
        head_dir = os.environ.get("HEAD_DIR", "/src/release")          # joint_head.safetensors + config
        backbone = os.environ.get("BACKBONE", "Qwen/Qwen3-0.6B")
        sys.path.insert(0, head_dir)
        import joint_schema_model as jsm
        self.jsm = jsm
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(snapshot_download(backbone))
        self.lm = AutoModelForCausalLM.from_pretrained(backbone, dtype=torch.bfloat16).to(self.device).eval()
        cfg = json.load(open(f"{head_dir}/joint_head_config.json"))
        self.head = jsm.JointSchemaHead(**cfg).to(self.device, torch.bfloat16).eval()
        self.head.load_state_dict(load_file(f"{head_dir}/joint_head.safetensors"))

    def predict(self, request: str = Input(description="Jev/SystemOne request body as JSON")) -> dict:
        req = json.loads(request)
        enc = self.jsm.encode_record(self.tok, {"state": req["state"], "questions": req["questions"]})
        ids = torch.tensor([enc.input_ids], device=self.device)
        with torch.no_grad():
            hs = self.lm.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).last_hidden_state
            logits = self.head(hs, ids, torch.ones_like(ids), [enc], self.lm.get_output_embeddings().weight)[0]
        answers = {}
        for q, lg in zip(enc.questions, logits):
            probs = dict(zip(q.option_ids, lg.float().softmax(-1).tolist()))
            answers[q.question_id] = self.jsm.systemone_answer(req["questions"][q.question_id], probs)
        self.record_metric("input_tokens", len(enc.input_ids))       # billing: COG_NEURON_COEFF_input_tokens
        return {"model": req.get("model", "miniclef"), "answers": answers, "usage": {"input_tokens": len(enc.input_ids)}}
