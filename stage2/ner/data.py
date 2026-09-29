"""Membaca dataset WNUT17 (format CoNLL) dan normalisasi token.

Dataset: satu token per baris, dipisah TAB dengan labelnya, baris kosong sebagai
pemisah kalimat. Di train, kalimat di dalam satu tweet juga dipisah oleh baris
yang hanya berisi tab. Kedua jenis pemisah diperlakukan sebagai batas kalimat.

Fungsi utama:
    read_conll(path) -> List[List[(token, label)]]
    load_splits(data_dir) -> dict split -> kalimat
    normalize(token) -> bentuk ternormalisasi untuk fitur leksikal / lookup embedding
"""
from __future__ import annotations

import html
import re
import unicodedata
from pathlib import Path

# Lokasi default: file .conll ada di folder Assignment 2 (parent dari stage2/)
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2]
SPLIT_FILES = {
    "train": "wnut17train.conll",
    "dev": "emerging.dev.conll",
    "test": "emerging.test.annotated",  # test gold (satu label per token)
}
ENTITY_TYPES = ["person", "location", "corporation", "product", "creative-work", "group"]
BIO_LABELS = ["O"] + [f"{p}-{t}" for t in ENTITY_TYPES for p in ("B", "I")]


def read_conll(path: str | Path) -> list[list[tuple[str, str]]]:
    """Kembalikan daftar kalimat; tiap kalimat daftar pasangan (token, label)."""
    sents: list[list[tuple[str, str]]] = []
    cur: list[tuple[str, str]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if stripped:  # baris berisi token (atau tab-only -> strip() jadi kosong -> pemisah)
                parts = line.rstrip("\r\n").split("\t")
                if len(parts) >= 2 and parts[0] != "":
                    cur.append((parts[0], parts[-1]))
                elif cur:  # baris tab-only: pemisah kalimat di train
                    sents.append(cur)
                    cur = []
            elif cur:  # baris kosong: pemisah kalimat/dokumen
                sents.append(cur)
                cur = []
    if cur:
        sents.append(cur)
    return sents


def load_splits(data_dir: str | Path = DEFAULT_DATA_DIR) -> dict[str, list[list[tuple[str, str]]]]:
    data_dir = Path(data_dir)
    return {split: read_conll(data_dir / fname) for split, fname in SPLIT_FILES.items()}


# ------------------------- Normalisasi token -------------------------
_URL_RE = re.compile(r"(?i)^(https?://|www\.)|\.com$|\.com/")
_MENTION_RE = re.compile(r"^@\w+$")
_DIGIT_RE = re.compile(r"\d")
_REPEAT_RE = re.compile(r"(.)\1{2,}")  # 3+ karakter berulang


def is_url(tok: str) -> bool:
    return bool(_URL_RE.search(tok)) or tok == "URL"


def is_mention(tok: str) -> bool:
    return bool(_MENTION_RE.match(tok))


def is_hashtag(tok: str) -> bool:
    return tok.startswith("#") and len(tok) > 1


def is_emoji(tok: str) -> bool:
    return any(unicodedata.category(c) == "So" or ord(c) >= 0x1F000 for c in tok)


def normalize(token: str) -> str:
    """Bentuk ternormalisasi untuk fitur/embedding (token asli tetap dipakai untuk fitur bentuk).

    Aturan: NFKC -> placeholder URL/@user -> ringkas huruf berulang -> digit jadi 0 -> lowercase.
    """
    if is_url(token):
        return "<URL>"
    if is_mention(token):
        return "<USER>"
    t = unicodedata.normalize("NFKC", token)
    t = html.unescape(t)
    t = _REPEAT_RE.sub(r"\1\1\1", t)  # sooooo -> sooo
    t = _DIGIT_RE.sub("0", t)  # 2017 -> 0000
    return t.lower()


def word_shape(token: str) -> str:
    """Ringkasan bentuk: X huruf besar, x huruf kecil, d digit, lainnya apa adanya."""
    out = []
    for c in token:
        if c.isupper():
            out.append("X")
        elif c.islower():
            out.append("x")
        elif c.isdigit():
            out.append("d")
        else:
            out.append(c)
    return "".join(out)


def short_shape(token: str) -> str:
    """Bentuk ringkas: hilangkan pengulangan berturut-turut (Xxxxx -> Xx)."""
    return re.sub(r"(.)\1+", r"\1", word_shape(token))


if __name__ == "__main__":
    data = load_splits()
    for split, sents in data.items():
        n_tok = sum(len(s) for s in sents)
        n_ent = sum(lab.startswith("B-") for s in sents for _, lab in s)
        print(f"{split:5s}: {len(sents):5d} kalimat, {n_tok:6d} token, {n_ent:5d} entitas")
