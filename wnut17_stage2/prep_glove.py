import gzip, os, numpy as np, re, unicodedata, html, pickle
GLOVE_GZ = os.environ.get('GLOVE_GZ', '/home/claude/glove-twitter-200.gz')  # glove-twitter-200.gz from the gensim-data release
from wnut_data import load

_URL_PREFIX = re.compile(r"^(https?://|www\.)", re.I)
_BARE_DOMAIN = re.compile(r"^\S+\.(com|net|org|io|co|ly)(/\S*)?$", re.I)   # Stage 1 rule: "*.com" becomes <URL>

def normalize(tok):
    """Normalized (lookup/feature) form of a token, following the Stage 1 plan (slides 21, report Table 8).
    The original token is always kept elsewhere; this form is only used for embedding lookup and lexical features.
    Order: HTML unescape, NFKC, URL/@user, hashtag body, digits to 0, cap repeated characters, lowercase."""
    t = html.unescape(tok)                       # "&amp;" -> "&"  (train has it as ONE token, test splits it)
    t = unicodedata.normalize("NFKC", t)         # unify look-alike characters
    if _URL_PREFIX.match(t) or _BARE_DOMAIN.match(t): return "<URL>"
    if t.startswith("@") and len(t) > 1: return "<USER>"
    if t.startswith("#") and len(t) > 1: t = t[1:]   # hashtag: drop '#', keep the word behind it (the is_hashtag flag is kept as a feature)
    t = re.sub(r"\d", "0", t)                    # every digit -> 0 (2017 -> 0000)
    t = re.sub(r"([^\d])\1{3,}", r"\1\1\1", t)   # cap a repeated character at 3 (soooooo -> sooo). BUG FIX: digit runs are NOT capped,
    return t.lower()                             # otherwise "2017" became "000" instead of "0000"

def glove_keys(tok):
    n = normalize(tok)
    keys = [tok.lower(), n]
    if n == "<URL>": keys = ["<url>"]
    elif n == "<USER>": keys = ["<user>"]
    elif re.fullmatch(r"0+([.,]0+)?", n): keys.append("<number>")
    if tok.startswith("#") and len(tok) > 1: keys.append(tok[1:].lower())
    return keys

if __name__ == "__main__":
    need = set()
    for s in ["train", "dev", "test"]:
        for x in load(s):
            for t in x["tokens"]:
                need.update(glove_keys(t))
    print("kunci dibutuhkan:", len(need))
    words, vecs, top_words, top_vecs = {}, [], [], []
    with gzip.open(GLOVE_GZ, "rt", encoding="utf-8", errors="ignore") as f:
        header = f.readline()
        print("header:", header.strip())
        for i, line in enumerate(f):
            w, rest = line.split(" ", 1)
            if w in need or i < 100000:
                v = np.array(rest.split(), dtype=np.float32)
                if v.shape[0] != 200: continue
                if w in need and w not in words:
                    words[w] = len(vecs); vecs.append(v)
                if i < 100000:
                    top_words.append(w); top_vecs.append(v)
    print("ditemukan:", len(words), "dari", len(need), "; top:", len(top_words))
    np.savez_compressed("glove_sub.npz", need_vecs=np.stack(vecs), top_vecs=np.stack(top_vecs))
    pickle.dump({"need_words": words, "top_words": top_words}, open("glove_vocab.pkl", "wb"))
