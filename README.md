# Assignment 14: Headline–Article Semantic Consistency

**Group 17**  
Anurag Rai — MT/AI/10016/26  
Ankit Singh — MT/AI/10022/26

This project compares two approaches for deciding whether a news headline is semantically consistent with its article:

1. **TF-IDF + cosine similarity** (traditional text-similarity baseline)
2. **Sentence embeddings + cosine similarity** using `sentence-transformers/all-MiniLM-L6-v2`

It also evaluates both methods on a manually reviewed challenge set containing deliberately modified headlines.

## Research task

Given a headline and its corresponding article text, assign one of two labels:

- `consistent` (1): the article supports the headline
- `inconsistent` (0): the article does not support the headline

The original FNC-1 stance labels are mapped as follows:

| FNC-1 stance | Project label |
|---|---|
| `agree`, `discuss` | `consistent` (1) |
| `disagree`, `unrelated` | `inconsistent` (0) |

## Repository layout

```text
NEW/
├── 01_Data_Splitting.ipynb
├── 02_Model_Training_Evaluation.ipynb
├── 03_Unseen_Modified_Challenge.ipynb
├── Assignment_14_Group_17_8_Slides.pptx
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── processed/       # Clean labeled source CSVs (local; not committed)
│   ├── splits/          # Fit, validation, and official-test CSVs (local)
│   └── challenge/       # Generated/reviewed challenge CSVs (local)
├── artifacts/           # Fitted TF-IDF vectorizer and related files (local)
└── results/
    ├── metrics/         # Exported aggregate metrics, when organized here
    ├── predictions/     # Row-level outputs (local/not committed)
    └── figures/          # Plots for the report and presentation
```

The current notebooks write some outputs directly under `results/`; this README reflects the project’s intended organization. Data files and fitted artifacts are excluded from Git by default because they can be large and may contain third-party news text.

## Data and split design

The processed training data contains 49,972 labeled pairs before the challenge article groups are reserved. The notebooks create these partitions:

| Partition | Rows | Purpose |
|---|---:|---|
| Fit | 31,957 | Fit the TF-IDF vocabulary and prepare model inputs |
| Validation | 7,670 | Select the decision threshold for each method |
| Official test | 25,413 | Final evaluation using the preserved official FNC-1 test split |
| Modified-headline challenge | 500 candidates before review | 350 original headlines plus 150 proposed numeric edits |

The challenge review approved 132 edited headlines and rejected 18. The scored challenge therefore contains 350 original examples and 132 approved edited examples: **482 rows**. The split notebook checks `body_id` overlap between fit, validation, official-test, and challenge partitions.

The original cleaned source files expected by the split notebook are `train_labeled_clean.csv` and `official_test_labeled_clean.csv`. In the current setup, `01_Data_Splitting.ipynb` reads them from:

```text
B:\MTech\Codes\SEM 1\MAI\ASS\Dataset\data\processed
```

If you clone or move the repository, update `source_dir` in the first configuration cell of `01_Data_Splitting.ipynb` to the local location of those files. The model and challenge notebooks also define the project root in their configuration cells; update `new_dir` if the repository is moved.

## Methods

### TF-IDF + cosine similarity

The headline and article model input are represented with word unigrams and bigrams. The TF-IDF vectorizer is fitted on the fit partition only. Cosine similarity between each headline vector and article vector is used as the score.

### Sentence embeddings + cosine similarity

The pretrained `sentence-transformers/all-MiniLM-L6-v2` encoder produces normalized headline and article embeddings. The encoder is frozen; no supervised fine-tuning is performed. Cosine similarity between the embeddings is the score.

### Thresholds and decision rule

Each method uses a threshold selected on validation data by maximizing Macro-F1. Ties are broken using F1 for the `consistent` class. A score greater than or equal to the threshold predicts `consistent`; a lower score predicts `inconsistent`.

The saved validation thresholds are approximately:

- TF-IDF: `0.0308`
- MiniLM: `0.4118`

The thresholds are fixed before official-test evaluation. ROC-AUC is calculated from continuous similarity scores rather than thresholded predictions.

## Evaluation

The official test and modified-headline challenge report:

- Accuracy
- Precision (`consistent` is the positive class)
- Recall (`consistent` is the positive class)
- Macro-F1
- ROC-AUC

The challenge also reports edited-headline rejection rate and paired original/edit flip success. These targeted measures show whether a model accepts an original headline but rejects its edited counterpart.

### Recorded official-test results

| Method | Accuracy | Precision | Recall | Macro-F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| TF-IDF + cosine | 0.9004 | 0.7457 | 0.9144 | 0.8762 | 0.9477 |
| MiniLM embeddings + cosine | 0.9571 | 0.8797 | 0.9601 | 0.9445 | 0.9848 |

These are the results recorded by the current project notebooks; rerunning may vary if package/model versions or source data change.

## Run the notebooks

Use Python 3.10 or later (the project was developed with Python 3.12) and run the notebooks in this order:

1. Install the packages listed in `requirements.txt`.
2. Place the two cleaned source CSVs in the configured source folder and verify the path in `01_Data_Splitting.ipynb`.
3. Run `01_Data_Splitting.ipynb` to generate the partitions and challenge review CSV.
4. Review each proposed edit against its article. Complete the review CSV before running the challenge notebook.
5. Run `02_Model_Training_Evaluation.ipynb` to prepare both similarity methods, select validation thresholds, and evaluate the official test. This saves the vectorizer and threshold values used by the challenge notebook.
6. Run `03_Unseen_Modified_Challenge.ipynb` to score the approved challenge examples and save challenge metrics.

The challenge notebook depends on the vectorizer and validation thresholds created by the model-training notebook, as well as the completed manual-review CSV.

Install dependencies from the repository root:

```bash
python -m pip install -r requirements.txt
```

Then launch Jupyter:

```bash
jupyter lab
```

Run the notebook cells from top to bottom. The MiniLM checkpoint is downloaded from Hugging Face on first use, so an internet connection is needed unless the model is already cached locally.

## Reproducibility and limitations

- The split uses a fixed random seed in the notebook and groups by `body_id` to reduce article-level leakage.
- Keep the official test partition separate from fitting and threshold selection.
- The challenge edits are numeric edits reviewed by a human; they do not represent all types of misleading headlines.
- Challenge articles are held out by article group from fit and validation, but are drawn from FNC-1’s training-side data. This is not a cross-news-source test.
- Cosine similarity measures lexical or semantic relatedness; it does not itself verify factual entailment.
- Do not claim that the models reliably detect factual changes based only on their official-test scores.

## Data handling

Do not commit raw or processed news datasets, row-level prediction files, fitted model files, or the manually reviewed challenge CSV unless you have checked the dataset’s license and have a reason to distribute them. The `.gitignore` keeps these files local by default. Aggregate metric tables, figures, and notebooks can be committed after checking that notebook outputs do not embed full article text.
