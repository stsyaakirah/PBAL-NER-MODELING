"""Opsi 1 (contextual): extract FROZEN RoBERTa-base token embeddings for WNUT17.

For each official split, run whole sentences through roberta-base (context matters),
align BPE subwords back to CoNLL tokens with word_ids(), mean-pool the last hidden
state per token, and cache results/roberta_{split}.pkl (list of [n_tok, 768] float32).
Nothing is fine-tuned. Used as an extra frozen input to the Bi-LSTM CRF.
"""
import os, pickle, numpy as np, torch
from transformers import AutoTokenizer, AutoModel
from wnut_data import load

torch.set_num_threads(os.cpu_count() or 4)
NAME = "roberta-base"
tok = AutoTokenizer.from_pretrained(NAME, add_prefix_space=True)
model = AutoModel.from_pretrained(NAME); model.eval()
DIM = model.config.hidden_size  # 768

@torch.no_grad()
def embed_split(seqs, bs=16):
    out = []
    for s in range(0, len(seqs), bs):
        batch = [x["tokens"] for x in seqs[s:s + bs]]
        enc = tok(batch, is_split_into_words=True, return_tensors="pt",
                  padding=True, truncation=True, max_length=256)
        hs = model(**enc).last_hidden_state  # (B, L, 768)
        for bi in range(len(batch)):
            wids = enc.word_ids(batch_index=bi)
            n = len(batch[bi]); acc = np.zeros((n, DIM), np.float32); cnt = np.zeros(n, np.float32)
            for j, wid in enumerate(wids):
                if wid is None or wid >= n: continue
                acc[wid] += hs[bi, j].numpy(); cnt[wid] += 1
            cnt[cnt == 0] = 1
            out.append((acc / cnt[:, None]).astype(np.float32))
        if s % (bs * 10) == 0:
            print(f"  {s + len(batch)}/{len(seqs)}", flush=True)
    return out

for split in ("train", "dev", "test"):
    path = f"results/roberta_{split}.pkl"
    if os.path.exists(path):
        print("skip (cached):", split, flush=True); continue
    seqs = load(split)
    print("embedding", split, len(seqs), "sentences", flush=True)
    vecs = embed_split(seqs)
    # sanity: token counts line up
    assert all(len(v) == len(x["tokens"]) for v, x in zip(vecs, seqs)), f"alignment mismatch in {split}"
    pickle.dump(vecs, open(path, "wb"))
    print("saved", path, "| dim", DIM, flush=True)
print("RoBERTa extraction done.", flush=True)
