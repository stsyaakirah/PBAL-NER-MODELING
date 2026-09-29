# Project Plan (Merged): NER on the WNUT2017 Emerging Entities Dataset

Assignment #2, Part 1. NLP course, sequence labeling.

This is the combined group plan. It joins the detailed data-analysis and preprocessing work (Sections 2 and 3) with the modeling and evaluation plan (Sections 4 to 8). It targets the WNUT2017 Novel and Emerging Entity Recognition dataset from Derczynski et al. 2017 (Proceedings of the 3rd Workshop on Noisy User-generated Text, pages 140 to 147, ACL, W17-4418).

## 1. Goal

Build and evaluate a NER system for the WNUT2017 shared task: find and type novel, rare, and emerging named entities in noisy user-generated text (tweets, Reddit, YouTube, and StackExchange comments).

The dataset is deliberately hard. Entity surface forms in training were blacklisted from dev and test, so systems that memorize frequent entities do badly and the task rewards generalization. The best shared-task system reached about 42 F1, so low scores are expected and normal.

## 2. Dataset

Six entity types: person, location (includes GPE and facility), corporation, product, creative-work, group. This is more types than the four-type PER/LOC/ORG/GPE scheme in the slides, and the boundaries between corporation, group, and product are genuinely fuzzy. The paper reports corporation and creative-work as the hardest classes.

### Splits

| | Train | Dev | Test |
|---|---|---|---|
| Source | Twitter (Ritter et al. 2011, via WNUT15) | YouTube | Reddit + Twitter + StackExchange |
| Documents | 1,000 tweets | 1,008 | 1,287 |
| Tokens | 65,124 (paper) | 15,734 | 23,394 |

The training data was inherited from the WNUT15 task, not created by the paper's authors. Dev and test (2,295 texts) are the paper's new contribution.

### Entity counts, recomputed from the released files

We counted directly from the files rather than trusting the paper's Table 1. The released `emerging.test.annotated` has more entities than the paper reported; the paper's counts are slightly stale. We use the file counts.

| Type | Dev | Test | % of test |
|---|---|---|---|
| person | 470 | 429 | 39.8 |
| location | 74 | 150 | 13.9 |
| corporation | 34 | 66 | 6.1 |
| product | 114 | 127 | 11.8 |
| creative-work | 105 | 142 | 13.2 |
| group | 39 | 165 | 15.3 |
| Total | 836 | 1,079 | 100 |

(Paper Table 1 for comparison: 835 dev, 1,040 test. Token and document counts match the paper exactly, confirming the files are complete.)

Two things to keep in the write-up:
- **Heavy class imbalance.** person dominates; corporation is the smallest class. A model can score well overall while failing on rare classes, so per-class scores matter.
- **Dev and test distributions differ on purpose.** Dev is all YouTube; test is a mix. The paper calls this a maximally adverse setup, so a dev-to-test drop is expected, not a bug.

## 3. Data preprocessing plan

This section is the preprocessing design (adapted from the group's data-analysis document).

### 3.1 Tokenization
- Use the tokenization already in the released CoNLL files. The text was tokenized with twokenizer (O'Connor et al. 2010) via GATE, so we do not retokenize; retokenizing would shift labels and break BIO alignment.
- If any text needs tokenizing from scratch, use ark-twokenize-py to match the original segmentation.
- Keep punctuation and special characters as separate tokens rather than dropping them: emoji, hashtags (#topic), mentions (@user), URLs, and expressive punctuation (?!, ..) can be part of entity boundaries or important context.

### 3.2 Special characters and encoding
- Read all files as UTF-8 (Reddit and StackExchange carry non-ASCII, emoji, and smart quotes).
- Minimal normalization only. Lowercasing is used as an extra feature, not as a replacement for the original token, because capitalization is a strong NER signal.
- Optionally map URL to `<URL>` and mention to `<USER>` as features, not as token replacements.
- Apply Unicode NFC normalization to standardize accent and emoji variants.

### 3.3 Vocabulary and embeddings
- Build the vocabulary from training data with an `<UNK>` token for out-of-vocabulary words. OOV rate will be high here because the task centers on novel and unseen entities.
- Recommended pretrained embeddings: GloVe Twitter (best domain match) and FastText (subword-level, which helps generalize to rare and emerging entities). Word2Vec Google News is not recommended (formal newswire domain).
- For non-neural models (CRF, SVM), embeddings can be added as numeric features, not as the main input.

### 3.4 Tagging scheme
- Keep IOB2 (BIO) as released. With 6 types this is 13 labels: O plus B and I for each type. All 13 were confirmed present in the files.
- Optional: convert to BIOES for a BiLSTM-CRF, since explicit end and single-token tags often help boundary detection.

### 3.5 Feature engineering (for the CRF and SVM track)

| Category | Features | Numeric form |
|---|---|---|
| Word-level | current word, previous/next word, context bigrams | dict feature (CRF) / one-hot via DictVectorizer (SVM) |
| Case | is_capitalized, is_all_upper, is_title_case, is_mixed_case | boolean |
| Character/morphology | prefix and suffix (1 to 4 chars), contains_digit, contains_hyphen, contains_punctuation | one-hot per n-gram |
| Twitter-specific | is_hashtag, is_mention, is_url, is_emoji | boolean |
| POS tag | from NLTK / Stanford CoreNLP (GATE Twitter model) | one-hot |
| Gazetteer (optional) | match against common name/city/brand lists; low expected benefit since the task targets novel entities | boolean |
| Context window | features combined over a plus/minus 2 token window | concatenated vector |
| Embedding (optional) | GloVe/FastText vector of the current token | continuous numeric |

Implementation notes: CRF uses per-token dictionary features (sklearn-crfsuite style); SVM one-hot encodes categorical features via DictVectorizer into a sparse matrix.

### 3.6 Data splitting
- Keep the official shared-task split; do not create a new random split. Dev is for tuning, test is for final evaluation only.
- The train-to-dev/test surface-form blacklist was applied by the authors, so there is no entity leakage across splits.
- If extra validation is needed while tuning CRF/SVM features, use k-fold cross-validation inside the training set only, never touching dev or test.

## 4. Models

Baseline first, then two model tracks, matching the slides' advice to beat an honest baseline.

### 4.1 Baseline: most-frequent-tag lookup
Store each training token's most common BIO tag; assign it at test time; unknown tokens get O. Expected to score very low here because test entities were blacklisted from training. We report it to make that point concrete.

### 4.2 Track A: CRF / SVM with the features in Section 3.5
A classic sequence-labeling approach well matched to the course (HMM/CRF material). CRF with sklearn-crfsuite is the primary model here; SVM with one-hot features is a comparison point. This track uses the hand-crafted features and is cheap to train without a GPU.

### 4.3 Track B: fine-tuned Transformer for token classification
A pretrained encoder with a token classification head, fine-tuned on the BIO data. Same family of method as the course paper (Yazid and Winarko 2023, IndoBERT for POS), adapted to English noisy text.
- Checkpoint choice: a tweet-pretrained model should help. Candidates: BERTweet, general `bert-base-cased` as a reference, or RoBERTa. Confirm exact Hugging Face names when loading (unverified until then).
- Subword alignment: label the first subword of each word, set other subwords to -100 (ignored), read predictions back from the first subword. Getting this wrong silently lowers scores, so add a unit check.

The strongest shared-task systems added external signals (gazetteers, tweet-trained embeddings, multi-task learning). Noted as options for later, not committed now.

## 5. Evaluation

The WNUT2017 task uses two entity-level metrics; we report both and label which is which. Both are entity-level, not token-level.

1. **Entity F1**: standard exact-span match (boundaries and type must both match). Compute with `seqeval`.
2. **Surface F1**: scores whether each unique (surface form, type) pair was captured at least once, so repeated entities count once and diverse recognition is rewarded. This is specific to this task and must be explained in the write-up.

No official scorer ships with the data, so we implement surface F1 ourselves (collect unique gold surface-form/type pairs, do the same for predictions, compute P/R/F over the sets). This is a short, hand-checkable calculation.

**Validating the scorer.** The `submissions/` folder holds the 7 shared-task teams' predictions on this exact test set, aligned to `emerging.test.annotated`. We score those with our code and check we reproduce the paper's Table 2 (for example UH-RiTUAL at 41.86 entity F1 and 40.24 surface F1). If our numbers match, the scorer is trustworthy. Note: `arcada.txt` is missing its predicted-tag column and needs cleanup; the other six are clean.

We also report token-level P/R/F1 (scikit-learn, excluding O) as a labeled internal diagnostic, never mixed with the two official entity-level scores. We report per-class scores because of the class imbalance in Section 2.

Per the course rules, for one small example we compute entity-level P/R/F1 by hand first, then confirm with the scorer, and report both.

## 6. Tools and reproducibility
- Python 3, fixed random seeds.
- Data: standard library and pandas; data is already tokenized and tagged.
- Models: sklearn-crfsuite for CRF, scikit-learn for SVM and the token-level diagnostic; Hugging Face `transformers` and `datasets` for the Transformer; `seqeval` for entity F1; our own hand-checked script for surface F1 (validated against the submissions).
- Ship a `requirements.txt` and a short README with exact run commands.
- Hardware: the CRF/SVM track runs without a GPU. The Transformer track needs a GPU; if none is available locally, use Google Colab or a reduced run rather than silently shrinking the experiment.

## 7. Risks and open questions
- **Class imbalance.** Rare classes may get near-zero recall. Report per-class scores; consider class weighting later.
- **Built-in distribution shift.** Expect a dev-to-test gap and explain it.
- **Surface F1 has no official scorer.** We implement and validate against the included submissions (Section 5).
- **Checkpoint names unverified** until loaded. Data files, format, tag set, and counts are already confirmed.
- **Low scores are normal** (best published ≈ 42 F1). Do not compare against the slides' 97% POS accuracy, which is token-level on an easier task.
- Open question for the lecturer: reproduce the full shared-task setup (train on the 2011 Twitter set, evaluate on the WNUT2017 test set with both metrics), or a simpler single-corpus exercise? This plan assumes the former.

## 8. Timeline (indicative)

| Stage | Work | Output |
|---|---|---|
| Week 1 | Load CoNLL files, verify counts, build preprocessing (Section 3) and the label map | Data loader, feature pipeline |
| Week 2 | Baseline + CRF/SVM track + full evaluation harness (entity F1, surface F1, token diagnostic) | Track A scores, hand-checked example |
| Week 3 | Fine-tune Transformer(s) | Track B scores, comparison table |
| Week 4 | Per-class error analysis, write-up, reproducibility check | Report, README, requirements.txt |

## 9. What success looks like
A working pipeline from raw CoNLL to predictions, a baseline that shows why memorization fails, a CRF/SVM model and at least one fine-tuned Transformer, and a results table with entity F1 and surface F1 overall and per class, plus an honest error analysis of the hard classes (corporation, creative-work) and the rare, unseen entities the dataset is built around.
