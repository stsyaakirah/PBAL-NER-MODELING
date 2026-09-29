"""Ekstraksi fitur per token untuk CRF dan SVM (jendela konteks +/-2 token).

token2features menghasilkan dict fitur per token. CRF (sklearn-crfsuite) memakai
dict ini langsung. SVM mengubahnya menjadi vektor sparse lewat DictVectorizer.

POS tag opsional (butuh nltk averaged_perceptron_tagger). Jika tidak tersedia,
fitur POS dilewati agar pipeline tetap jalan.
"""
from __future__ import annotations

from .data import (is_emoji, is_hashtag, is_mention, is_url, normalize,
                   short_shape, word_shape)

try:
    import nltk

    nltk.pos_tag(["test"])
    _POS_OK = True
except Exception:
    _POS_OK = False


def _word_feats(tok: str, prefix: str) -> dict:
    """Fitur untuk satu token pada posisi relatif (prefix '', '-1:', '+1:', ...)."""
    return {
        f"{prefix}norm": normalize(tok),
        f"{prefix}lower": tok.lower(),
        f"{prefix}shape": short_shape(tok),
        f"{prefix}istitle": tok.istitle(),
        f"{prefix}isupper": tok.isupper(),
        f"{prefix}isdigit": tok.isdigit(),
        f"{prefix}is_url": is_url(tok),
        f"{prefix}is_mention": is_mention(tok),
        f"{prefix}is_hashtag": is_hashtag(tok),
        f"{prefix}is_emoji": is_emoji(tok),
    }


def token2features(tokens: list[str], i: int, pos_tags: list[str] | None = None) -> dict:
    tok = tokens[i]
    f = {
        "bias": 1.0,
        "word.full_shape": word_shape(tok),
        "word.len": len(tok),
        "word.has_digit": any(c.isdigit() for c in tok),
        "word.has_hyphen": "-" in tok,
        "word.has_apos": "'" in tok,
        "word.isalpha": tok.isalpha(),
    }
    f.update(_word_feats(tok, ""))
    # affix karakter 1-4
    low = tok.lower()
    for k in range(1, 5):
        if len(low) >= k:
            f[f"pre{k}"] = low[:k]
            f[f"suf{k}"] = low[-k:]
    # konteks +/-2
    for d in (-2, -1, 1, 2):
        j = i + d
        pre = f"{d:+d}:"
        if 0 <= j < len(tokens):
            f.update(_word_feats(tokens[j], pre))
        else:
            f[f"{pre}BOS_EOS"] = True
    # bigram konteks langsung
    if i > 0:
        f["-1:bigram"] = tokens[i - 1].lower() + "|" + low
    # POS (opsional)
    if pos_tags is not None:
        f["pos"] = pos_tags[i]
        if i > 0:
            f["-1:pos"] = pos_tags[i - 1]
        if i + 1 < len(tokens):
            f["+1:pos"] = pos_tags[i + 1]
    return f


def sent2features(sent_tokens: list[str], use_pos: bool = True) -> list[dict]:
    pos = None
    if use_pos and _POS_OK:
        pos = [t for _, t in nltk.pos_tag(sent_tokens)]
    return [token2features(sent_tokens, i, pos) for i in range(len(sent_tokens))]


def dataset_features(sents, use_pos: bool = True):
    """sents: list kalimat [(tok, lab)]. Kembalikan (X list-of-list-dict, y list-of-list-label)."""
    X, y, toks = [], [], []
    for sent in sents:
        t = [tok for tok, _ in sent]
        X.append(sent2features(t, use_pos))
        y.append([lab for _, lab in sent])
        toks.append(t)
    return X, y, toks


POS_AVAILABLE = _POS_OK
