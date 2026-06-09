# ATRS

Official implementation of:
> Lim, H., Li, X., Park, S., Li, Q., & Kim, J. (2026).
**Reducing contextual noise in review-based recommendation via aspect term extraction and attention modeling**.
_Information Sciences_, 735, 123078. [Paper](https://doi.org/10.1016/j.ins.2026.123078)

## Overview
This repository is the official implementation of ATRS (Aspect Term-aware Recommender System), published in *Information Sciences* (2026).

Most review-based recommendation models process entire review bodies indiscriminately, allowing aspect-relevant signal to be diluted by surrounding context. ATRS addresses this by routing review text through a dedicated **Aspect Term Extraction (ATE)** stage that filters out non-aspect content before downstream encoding.

The retained aspect terms are encoded with a 1D-CNN over Word2Vec embeddings, fused with user/item ID embeddings, and passed through a self-attention block to form aspect-aware user and item representations. These are concatenated and forwarded to an MLP that predicts a continuous rating score as a regression target. Quantitative comparisons against representative recommendation baselines on Amazon and Yelp datasets are reported in [Experimental Results](#experimental-results).

## Repository Structure

```bash
├── data/
│   ├── raw/                        # Source datasets — place {fname}.{raw_ext} here
│   ├── processed/                  # Pipeline parquet caches (preprocessed / aspects)
│   └── ate_output/                 # PyABSA workspace + extraction JSON
│       └── .pyabsa/                # Contained pyabsa CWD: checkpoints/, checkpoints.json, result JSON
│
├── model/
│   ├── atrs.py                     # ATRS architecture, trainer, predictor
│   ├── ATRS Architecture.png       # Architecture diagram
│   └── save/                       # Best checkpoint per dataset (best.pth)
│
├── src/
│   ├── config.yaml                 # Single source of truth for all hyperparameters
│   ├── data_processing.py          # DataProcessor pipeline + Dataset/DataLoader factory
│   ├── aspect_extraction.py        # ATExtractor — PyABSA wrapper for aspect term extraction
│   ├── preprocessing.py            # Review-text cleaning and row filters
│   ├── path.py                     # Project path constants (auto-creates runtime folders)
│   └── utils.py                    # Generic helpers — I/O, metrics, seeding
│
├── main.py                         # Entry point: data preparation → train → test
├── requirements.txt
└── README.md
```

## Model Description

ATRS consists of two sequential modules. Aspect extraction runs in [`src/aspect_extraction.py`](src/aspect_extraction.py) (orchestrated by [`src/data_processing.py`](src/data_processing.py)); the recommender network is in [`model/atrs.py`](model/atrs.py). The full architecture is illustrated below.

<p align="center">
  <img src="model/ATRS Architecture.png" alt="ATRS Architecture" width="800">
</p>

### 1. Aspect Term Extraction Module
A pretrained Transformer encoder (PyABSA's English ATE checkpoint, FAST-LCF-ATEPC over DeBERTa-v3-base) reads each cleaned review and emits BIO-tagged aspect terms. Per-row aspect lists are then aggregated into per-user and per-item aspect sets, which become the inputs to the RS module.

### 2. Recommender System Module
Each user and item aspect set is tokenized over a Word2Vec-trained vocabulary, encoded by a 1D-CNN (`AspectEncoder`), and concatenated with a learned ID embedding. The fused vector is projected and passed through a multi-head self-attention + FFN block (`SelfAttentionBlock`) to yield aspect-aware user and item representations. Their concatenation is fed to an MLP regressor (`ATRS.regressor`) that outputs the predicted rating.

## How to Run

### Configuration
All hyperparameters live in [`src/config.yaml`](src/config.yaml) — it is the single source of truth. Defaults reproduce the paper experiments.

A CUDA-capable GPU is recommended; `main.py` falls back to CPU with a warning if CUDA is unavailable. See [`requirements.txt`](requirements.txt) for the GPU wheel and CPU-only setup.

End-to-end run:
```bash
conda create -n atrs python=3.11
conda activate atrs
pip install -r requirements.txt
python main.py
```

### Data Preparation
Place the dataset as `data/raw/{fname}.{raw_ext}` where `{fname}` and `{raw_ext}` match `data.fname` / `data.raw_ext` in `config.yaml`. The file is read as JSON-lines (one review object per line) — **each line must carry the columns below**, or the run aborts at load with a `KeyError`.

| Column | Role |
|---|---|
| `user_id` | Reviewer id — user-side aspect aggregation and ID embedding. |
| `parent_asin` | Product id — item-side aspect aggregation and ID embedding. |
| `text` | Review body — cleaned, then aspect terms are extracted from it (`review_text` is also accepted as an alias). |
| `rating` | Ground-truth rating; the regression target the model predicts. |
| `verified_purchase` | Boolean flag; only verified-purchase reviews are kept. |

Optional: an `aspect` column of pre-extracted per-row aspect lists — if present, the PyABSA extraction stage is skipped. Any other columns are ignored. The pipeline writes two cache layers under `data/processed/`:

- **`{fname}_preprocessed.parquet`** — written after text cleaning and the k-core filter.
  - Columns: the required columns above + `clean_text` (HTML/URL-stripped, lowercased, contraction-expanded, stop-word-removed, lemmatized review body). Any extra raw columns pass through untouched.
- **`{fname}_aspects.parquet`** — adds the extracted aspect terms and their per-user/item aggregation.
  - Columns: the preprocessed columns + `aspect` (per-row aspect-term list), `user_aspect_set` / `item_aspect_set` (each id's aspect terms flattened across all its reviews).

### Re-runs and caching
On every `python main.py`, the pipeline resumes from the most-complete cache on disk, checking newest-first (aspects → preprocessed → raw) and falling through to the next-earliest stage. The train/test split, Word2Vec, and sequence padding always run fresh in memory, so changes to `test_size`, `seed`, `val_ratio`, `aspect_length_percentile`, or `w2v_*` take effect on the next run. To re-trigger an upstream stage, delete its parquet.

## Experimental Results

ATRS was evaluated on three real-world review datasets: Musical Instruments, Video Games, and Yelp (Pennsylvania).
The results demonstrate that ATRS consistently outperforms representative baselines across all evaluation metrics, achieving average improvements of 19.54% in MAE and 11.89% in RMSE.

<div align="center">
  <table>
    <thead>
      <tr>
        <th rowspan="2">Model</th>
        <th colspan="4">Musical Instruments</th>
        <th colspan="4">Video Games</th>
        <th colspan="4">Yelp</th>
      </tr>
      <tr>
        <th>MAE</th>
        <th>MSE</th>
        <th>RMSE</th>
        <th>MAPE</th>
        <th>MAE</th>
        <th>MSE</th>
        <th>RMSE</th>
        <th>MAPE</th>
        <th>MAE</th>
        <th>MSE</th>
        <th>RMSE</th>
        <th>MAPE</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>PMF</td>
        <td>1.306</td><td>2.640</td><td>1.625</td><td>35.034</td>
        <td>1.220</td><td>2.407</td><td>1.551</td><td>33.948</td>
        <td>1.276</td><td>2.803</td><td>1.674</td><td>38.330</td>
      </tr>
      <tr>
        <td>NCF</td>
        <td>1.174</td><td>1.705</td><td>1.306</td><td>35.401</td>
        <td>0.948</td><td>1.331</td><td>1.154</td><td>35.032</td>
        <td>1.085</td><td>1.674</td><td>1.294</td><td>39.320</td>
      </tr>
      <tr>
        <td>DeepCoNN</td>
        <td>0.786</td><td>1.137</td><td>1.067</td><td>29.931</td>
        <td>0.847</td><td>1.263</td><td>1.124</td><td>32.850</td>
        <td>0.937</td><td>1.381</td><td>1.175</td><td>38.276</td>
      </tr>
      <tr>
        <td>NARRE</td>
        <td>0.767</td><td>0.993</td><td>0.997</td><td>29.459</td>
        <td>0.776</td><td>1.173</td><td>1.083</td><td>30.518</td>
        <td>0.886</td><td>1.212</td><td>1.101</td><td>36.724</td>
      </tr>
      <tr>
        <td>AENAR</td>
        <td>0.665</td><td>0.970</td><td>0.985</td><td>27.193</td>
        <td>0.693</td><td>1.002</td><td>1.001</td><td>28.039</td>
        <td>0.845</td><td>1.177</td><td>1.085</td><td>35.605</td>
      </tr>
      <tr>
        <td>SAFMR</td>
        <td>0.705</td><td>0.975</td><td>0.987</td><td>28.388</td>
        <td>0.711</td><td>1.033</td><td>1.016</td><td>30.016</td>
        <td>0.881</td><td>1.229</td><td>1.109</td><td>36.076</td>
      </tr>
      <tr>
        <td>MFNR</td>
        <td>0.708</td><td>0.965</td><td>0.982</td><td>26.922</td>
        <td>0.730</td><td>0.980</td><td>0.990</td><td>27.863</td>
        <td>0.855</td><td>1.174</td><td>1.084</td><td>33.923</td>
      </tr>
      <tr>
        <td><b>ATRS (Proposed)</b></td>
        <td><b>0.640</b></td><td><b>0.933</b></td><td><b>0.966</b></td><td><b>26.638</b></td>
        <td><b>0.646</b></td><td><b>0.970</b></td><td><b>0.985</b></td><td><b>27.537</b></td>
        <td><b>0.832</b></td><td><b>1.163</b></td><td><b>1.078</b></td><td><b>34.917</b></td>
      </tr>
    </tbody>
  </table>
</div>

## Citation

If you use this repository in your research, please cite:

```bibtex
@article{LIM2026123078,
  title = {Reducing contextual noise in review-based recommendation via aspect term extraction and attention modeling},
  author = {Heena Lim and Xinzhe Li and Seonu Park and Qinglong Li and Jaekyeong Kim},
  journal = {Information Sciences},
  volume = {735},
  pages = {123078},
  year = {2026},
  doi = {10.1016/j.ins.2026.123078}
}
```

## Contact

For research inquiries or collaborations, please contact:

**Seonu Park**
Ph.D. Student, Department of Big Data Analytics
Kyung Hee University
Email: sunu0087@khu.ac.kr

**Qinglong Li**
Assistant Professor, Division of Computer Engineering
Hansung University
Email: leecy@hansung.ac.kr

_Last updated: June 2026_
