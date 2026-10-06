# -*- coding: utf-8 -*-
"""ระบบวิเคราะห์ความสัมพันธ์เชิงเหตุและผล (Causal BIO Tagging) ภาษาไทย — Flask web UI
"""
import html
import os
import re

import torch
import torch.nn as nn
from flask import Flask, render_template, request
from transformers import AutoTokenizer, AutoModel
from torchcrf import CRF
from pythainlp.tokenize import word_tokenize

try:
    from safetensors.torch import load_file as load_safetensors
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False

LABELS = ["O", "B-CAUSE", "I-CAUSE", "B-EFFECT", "I-EFFECT"]
ID2LABEL = {i: l for i, l in enumerate(LABELS)}
BASE_MODEL_NAME = "airesearch/wangchanberta-base-att-spm-uncased"

MODEL_DIR = os.environ.get(
    "MODEL_DIR", 
    r"C:\Users\ideapad GAMING\Downloads\causal_flask_app\models\Causal"
)
MAX_LEN = 128


class WangchanBertaCRF(nn.Module):
    def __init__(self, model_name, num_labels, dropout_prob=0.1, freeze_layers=0):
        super().__init__()
        self.num_labels = num_labels
        self.roberta = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(self.roberta.config.hidden_size, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

    @staticmethod
    def _pack(emissions, labels, word_mask):
        B, _, C = emissions.shape
        counts = word_mask.sum(dim=1)
        max_words = max(int(counts.max().item()), 1)

        packed_em = emissions.new_zeros((B, max_words, C))
        packed_mask = torch.zeros((B, max_words), dtype=torch.bool, device=emissions.device)
        packed_tags = torch.zeros((B, max_words), dtype=torch.long, device=emissions.device)

        for b in range(B):
            idx = word_mask[b].nonzero(as_tuple=True)[0]
            n = int(idx.numel())
            if n == 0:
                packed_mask[b, 0] = True
                continue
            packed_em[b, :n] = emissions[b, idx]
            packed_mask[b, :n] = True
            if labels is not None:
                packed_tags[b, :n] = labels[b, idx]
        return packed_em, packed_mask, packed_tags

    def forward(self, input_ids, attention_mask, labels=None, word_mask=None):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs[0])
        emissions = self.classifier(sequence_output)

        if word_mask is None:
            word_mask = labels != -100

        packed = self._pack(emissions, labels, word_mask)
        if labels is not None:
            _, packed_mask, packed_tags = packed
            loss = -self.crf(packed[0], packed_tags, mask=packed_mask, reduction="mean")
            return loss, packed
        return packed

    def decode(self, packed):
        packed_em, packed_mask, _ = packed
        return self.crf.decode(packed_em, mask=packed_mask)

    @classmethod
    def from_pretrained(cls, checkpoint_dir, base_model_name, num_labels, **kwargs):
        model = cls(base_model_name, num_labels, **kwargs)
        safe_path = os.path.join(checkpoint_dir, "model.safetensors")
        bin_path = os.path.join(checkpoint_dir, "pytorch_model.bin")

        if os.path.isfile(safe_path) and HAS_SAFETENSORS:
            state_dict = load_safetensors(safe_path)
        elif os.path.isfile(bin_path):
            state_dict = torch.load(bin_path, map_location="cpu")
        else:
            raise FileNotFoundError(f"ไม่พบไฟล์น้ำหนักโมเดลที่ {checkpoint_dir}")

        # แปลงคีย์ gamma -> weight และ beta -> bias เพื่อรองรับสถาปัตยกรรม CamemBERT/WangchanBERTa
        mapped_state_dict = {}
        for key, val in state_dict.items():
            new_key = key
            if new_key.endswith(".gamma"):
                new_key = new_key[:-6] + ".weight"
            elif new_key.endswith(".beta"):
                new_key = new_key[:-5] + ".bias"
            mapped_state_dict[new_key] = val

        # โหลด state_dict แบบผ่อนปรน (strict=False)
        missing_keys, unexpected_keys = model.load_state_dict(mapped_state_dict, strict=False)
        print(f"โหลดน้ำหนักโมเดลเรียบร้อยแล้ว (Missing: {len(missing_keys)}, Unexpected: {len(unexpected_keys)})")
        return model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"กำลังใช้งานอุปกรณ์: {device}")

if not os.path.exists(MODEL_DIR):
    raise SystemExit(f"❌ ไม่พบโฟลเดอร์โมเดลที่ '{MODEL_DIR}' กรุณาตรวจสอบ Path")

print(f"กำลังโหลดโมเดลจาก '{MODEL_DIR}' ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = WangchanBertaCRF.from_pretrained(MODEL_DIR, BASE_MODEL_NAME, num_labels=len(LABELS)).to(device).eval()
print("โมเดลพร้อมใช้งาน!")


@torch.no_grad()
def analyze_text(raw_text: str) -> dict:
    if not raw_text.strip():
        return {"original_text": raw_text, "results": [], "error": "ข้อความว่างเปล่า"}

    words = word_tokenize(raw_text, engine="newmm")
    if not words:
        return {"original_text": raw_text, "results": [], "error": "ไม่สามารถตัดคำได้"}

    input_ids = [tokenizer.cls_token_id]
    word_mask = [False]

    for word in words:
        pieces = tokenizer.tokenize(word)
        if not pieces:
            pieces = [tokenizer.unk_token]
        piece_ids = tokenizer.convert_tokens_to_ids(pieces)
        input_ids.extend(piece_ids)
        word_mask.append(True)
        word_mask.extend([False] * (len(pieces) - 1))
        if len(input_ids) >= MAX_LEN - 1:
            break

    input_ids = input_ids[: MAX_LEN - 1] + [tokenizer.sep_token_id]
    word_mask = word_mask[: MAX_LEN - 1] + [False]

    input_ids_t = torch.tensor([input_ids], dtype=torch.long, device=device)
    attn_t = torch.tensor([[1] * len(input_ids)], dtype=torch.long, device=device)
    word_mask_t = torch.tensor([word_mask], dtype=torch.bool, device=device)

    packed = model(input_ids=input_ids_t, attention_mask=attn_t, word_mask=word_mask_t)
    pred_tag_ids = model.decode(packed)[0]

    valid_words = words[: len(pred_tag_ids)]
    pred_labels = [ID2LABEL[idx] for idx in pred_tag_ids]

    cause_words = [w for w, lbl in zip(valid_words, pred_labels) if "CAUSE" in lbl]
    effect_words = [w for w, lbl in zip(valid_words, pred_labels) if "EFFECT" in lbl]

    cause_text = "".join(cause_words) if cause_words else "-"
    effect_text = "".join(effect_words) if effect_words else "-"

    has_causal = (cause_text != "-") or (effect_text != "-")
    label_status = "พบความสัมพันธ์เชิงเหตุและผล" if has_causal else "ไม่พบความสัมพันธ์เชิงเหตุผล"

    result_item = {
        "label": label_status,
        "confidence": 1.0 if has_causal else 0.0,
        "low_confidence": False,
        "marker": "BIO Tagging",
        "cause": cause_text,
        "effect": effect_text,
    }

    return {"original_text": raw_text, "results": [result_item], "error": None}


app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    input_text = ""
    if request.method == "POST":
        input_text = request.form.get("review_text", "").strip()
        if input_text:
            result = analyze_text(input_text)
    return render_template("index.html", result=result, input_text=input_text)


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    data = request.get_json(silent=True) or {}
    text = str(data.get("text", "")).strip()
    if not text:
        return {"error": "กรุณาส่งข้อความในฟิลด์ 'text'"}, 400
    return analyze_text(text)


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)