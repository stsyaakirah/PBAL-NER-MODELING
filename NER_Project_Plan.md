# Project Plan: Named Entity Recognition on the WNUT2017 Emerging Entities Dataset

Assignment #2, Part 1. NLP course, sequence labeling.

This plan targets the WNUT2017 Novel and Emerging Entity Recognition dataset from Derczynski et al. 2017 (Proceedings of the 3rd Workshop on Noisy User-generated Text, pages 140 to 147).

## 1. Goal

Build and evaluate a named entity recognition (NER) system for the WNUT2017 shared task: find and type **novel, rare, and emerging** named entities in noisy user-generated text (tweets, Reddit, YouTube, and StackExchange comments).

The point of this dataset is deliberately hard. Systems that memorize frequent entities from training do badly here, because the test entities are mostly ones never seen in training. Success means generalizing, not memorizing. The best system in the shared task reached about 42 F1, so low scores are expected and normal.

## 2. Dataset

WNUT2017, from Derczynski et al. 2017. The numbers below are taken from the paper (Table 1 and Section 3.1), not invented.

### Entity types (6)

person, location (includes GPE and facility), corporation, product (tangible goods or well-defined services), creative-work (song, movie, book, and so on), group (bands, sports teams, non-corporate organizations).

This is more types than the four-type PER/LOC/ORG/GPE scheme in the course slides, and the boundaries between corporation, group, and product are genuinely fuzzy. The paper reports that corporation and creative-work were the hardest classes.

### Splits and size

| | Train | Dev | Test |
|---|---|---|---|
| Source | Twitter (Ritter et al. 2011) | YouTube | Twitter, Reddit, StackExchange |
| Documents | 1,000 tweets | 1,008 | 1,287 |
| Tokens | 65,124 | 15,734 | 23,394 |
| Entities | not stated in paper | 835 | 1,040 |

Per-type entity counts (dev / test), from Table 1:

| Type | Dev | Test |
|---|---|---|
| person | 470 | 414 |
| location | 74 | 139 |
| corporation | 34 | 70 |
| product | 114 | 127 |
| creative-work | 104 | 140 |
| group | 39 | 150 |

Two things to notice in this table and to keep in the write-up:
- **Heavy class imbalance.** person dominates the dev set, while corporation and group are rare. A model can get a decent overall score while failing on the rare classes, so per-class scores matter.
- **The split distributions differ on purpose.** Dev is all YouTube, test is a mix, and the type proportions shift between them (for example group jumps from 39 in dev to 150 in test). The paper calls this a maximally adverse setup. So dev performance will not cleanly predict test performance, and we should say so rather than treat a dev drop as a bug.

### A key design fact

Surface forms marked as entities in the training data were blacklisted from dev and test. So almost none of the test entities appear as entities in training. This is why plain lookup and memorization fail, and why the task rewards sub-word features, context, and external signals.

## 3. Data format and files

The data is already downloaded and sits in this assignment folder. It came from the official `emerging_entities_17` release by the task organizers (Leon Derczynski), distributed under CC-BY 4.0. The files we use:

| Role | File | Notes |
|---|---|---|
| Train | `wnut17train.conll` | The Ritter et al. 2011 Twitter set, redistributed by the task |
| Dev | `emerging.dev.conll` | YouTube, with gold labels |
| Test | `emerging.test.annotated` | Mixed sources, with gold labels; use this for scoring |
| Test (unlabeled) | `emerging.test.conll` | Same test text without labels, for generating predictions |

The release also includes two reference folders we do not train on: `source/` (raw and tokenized source text grouped by origin: Reddit, StackExchange, the four Twitter events, YouTube) for data provenance, and `submissions/` (the 7 shared-task teams' predictions on the test set) which we reuse to validate our scorer in Section 5.

The data is in **CoNLL format**: one token per line, a tab, then its BIO tag, with a blank line between sentences. This is already the tagging format we need, so there is no span-to-token conversion work. The text is pre-tokenized (twokenizer), so we reuse that tokenization rather than retokenizing.

BIO scheme: B for the first token of an entity, I for continuation, O for outside. With 6 types this is 2n+1 = 13 labels (B and I for each of the 6 types, plus O). All 13 labels were confirmed present in the files.

### Counts verified from the files

We recomputed token, entity, and segment counts directly from the files rather than trusting the paper's Table 1. Results:

| File | Tokens | Entities (B- tags) | Sentence segments |
|---|---|---|---|
| `wnut17train.conll` | 62,730 | 1,975 | 3,394 |
| `emerging.dev.conll` | 15,733 | 836 | 1,009 |
| `emerging.test.annotated` | 23,394 | 1,079 | 1,287 |

Two mismatches against the paper to note in the write-up, neither a data problem:
- **Entity counts differ slightly** from Table 1 (paper: 835 dev, 1,040 test; files: 836, 1,079). This is a counting-convention difference. The token counts match the paper almost exactly (15,734 dev, 23,394 test) and the document counts match (1,008 dev, 1,287 test), which confirms the files are the complete official release. We report our own recomputed numbers and say so.
- **Train tokens** are 62,730 here versus the paper's 65,124, a tokenization difference in the redistributed 2011 Twitter set. This is the official training file.

## 4. Models

Baseline first, then a stronger model, matching the slides' advice to beat an honest baseline.

### 4.1 Baseline: most-frequent-tag lookup
- Store each training token's most common BIO tag; assign it at test time; unknown tokens get O.
- This is expected to score very low here, close to zero on rare entities, precisely because the test entities were blacklisted from training. That is the intended lesson of the dataset: memorization does not transfer. We report it to make that point concrete, not because we expect it to be competitive.

### 4.2 Main model: fine-tuned Transformer for token classification
- A pretrained encoder with a token classification head, fine-tuned on the WNUT2017 BIO data. This is the same family of method as the course paper (Yazid and Winarko 2023), which fine-tuned IndoBERT for POS tagging, adapted here to English noisy text and entity BIO tags.
- **Checkpoint choice.** Because the text is noisy social media, a model pretrained on tweets should help. Candidates: BERTweet, or a general `bert-base-cased` as a reference point, or RoBERTa. We fine-tune the general model first as a floor, then a social-media model, and compare. Exact Hugging Face checkpoint names must be confirmed before use; treat them as unverified until loaded.
- **Subword alignment.** The encoder splits words into subwords. Standard rule: label the first subword of each word with the word's tag, set the other subwords to an ignored label (-100), and read predictions back from the first subword. Getting this wrong silently lowers scores, so it needs care and a unit check.

The strongest shared-task systems added external signals (gazetteers, word embeddings trained on large tweet collections, multi-task learning). We note these as options for Part 2 rather than committing to them now.

### 4.3 Optional middle model
A BiLSTM-CRF or CRF between the baseline and the Transformer, if time allows. Listed as optional to keep the plan realistic.

## 5. Evaluation

The WNUT2017 task uses **two entity-level metrics**, and we will report both and name which is which. Both are entity-level, not token-level.

1. **Entity F1** (called F1 entity in the paper). Standard exact-span match: an entity is correct only if its boundaries and type both match the gold entity. Precision, recall, F1.
2. **Surface F1** (called F1 surface). Scores whether each **unique surface form** that is an entity was captured at least once, rather than counting every occurrence. This rewards systems that catch a rare entity even once and does not let a few frequent entities inflate the score. This metric is specific to this task and worth explaining clearly in the write-up, since it is not the usual NER metric.

For reference, the paper's leaderboard (Table 2): best system UH-RiTUAL at 41.86 entity F1 and 40.24 surface F1; the field ran from about 26 to 42 entity F1. We use these only as context, and we will only compare our numbers to them after confirming we use the same metric definitions and the same test file.

We will also report **token-level** precision, recall, and F1 (scikit-learn, excluding O from the averages) as an internal diagnostic, and we will always label it as token-level so it is never confused with the two official entity-level scores. Per the course rules, high token accuracy can hide poor entity recognition, which is doubly true here.

We report **per-class** scores, not just the overall average, because of the class imbalance noted in Section 2. The official entity F1 can be computed with `seqeval`. The organizers' scoring script is **not** included in the data release, so we implement surface F1 ourselves: collect the set of unique entity surface forms per type in the gold data, do the same for the predictions, and compute precision, recall, and F1 over those sets. This is a short, hand-checkable calculation.

**Validating our scorer with the included submissions.** The `submissions/` folder holds the 7 shared-task teams' predictions on this exact test set, in the same CoNLL format, aligned to `emerging.test.annotated`. We score those files with our own code and check that we reproduce the paper's Table 2 (for example UH-RiTUAL at 41.86 entity F1 and 40.24 surface F1). If our numbers match, the scorer is trustworthy. This is a stronger check than hand-verifying one example alone. Note: `arcada.txt` looks to be missing its predicted-tag column and would need cleanup first; the other six submissions are clean.

Per the course working rules, for one small example we will compute entity-level precision, recall, and F1 by hand first, then confirm with the scorer, and report both. If they disagree we find out why before trusting the code.

## 6. Tools and reproducibility

- Python 3, fixed random seeds.
- Data handling: standard library and pandas. The data is already tokenized and tagged, so no heavy preprocessing.
- Modeling: Hugging Face `transformers` and `datasets` for the fine-tuned model; scikit-learn for the baseline and token-level diagnostic; `seqeval` for entity F1; our own hand-checked script for surface F1, validated against the included submissions (Section 5).
- Deliverables include a `requirements.txt` and a short README with the exact commands to reproduce each result.
- Hardware: fine-tuning needs a GPU. If no local GPU is available we say so and use Google Colab, or run a reduced version rather than silently shrinking the experiment.

## 7. Risks and open questions

- **Class imbalance.** Rare classes (corporation, group in dev) may get near-zero recall. Mitigation: report per-class scores and consider class weighting in Part 2. Do not hide weak classes behind the average.
- **Train and test distribution shift is built in.** Dev is YouTube only, test is a different mix. Expect a gap and explain it rather than treating it as a failure.
- **Surface F1 is an unusual metric, and no official scorer ships with the data.** We implement it ourselves and validate against the paper's Table 2 using the included `submissions/` files (Section 5), which removes the need for the organizers' original script.
- **Checkpoint names are unverified** until loaded. The data files, format, tag set, and counts are already confirmed (Section 3), so that risk is closed; only the Hugging Face checkpoint names still need confirming when we load them.
- **Low scores are normal.** The best published system is around 42 F1. We frame our numbers against that reality, not against the slides' 97 percent POS figure, which is token accuracy on a different and easier task.
- Open question for the lecturer: is the goal to reproduce the shared-task setup (train on the 2011 Twitter set, evaluate on the WNUT2017 test set with entity and surface F1), or a simpler single-corpus NER exercise? This plan assumes the former.

## 8. Timeline (indicative)

| Stage | Work | Output |
|---|---|---|
| Week 1 | Load the confirmed CoNLL files into a datasets pipeline, write the label map, sanity-check a few sentences | Working data loader |
| Week 2 | Baseline model and full evaluation harness (entity F1, surface F1, token-level diagnostic) | Baseline scores, hand-checked example |
| Week 3 | Fine-tune a general and a social-media Transformer | Model scores, comparison table |
| Week 4 | Per-class error analysis, write-up, reproducibility check | Report, README, requirements.txt |

## 9. What success looks like

A working pipeline that reads the WNUT2017 CoNLL data, a baseline to show why memorization fails on emerging entities, at least one fine-tuned Transformer, and a clear results table with entity F1 and surface F1 overall and per class, plus an honest error analysis of the hard classes (corporation, creative-work) and of the rare, unseen entities the dataset is built around.
