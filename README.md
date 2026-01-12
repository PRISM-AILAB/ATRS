# ATRS

Official implementation of the paper:
> **Lim, H., Li, X., Park, S., Li, Q., & Kim, J. (2026). Reducing contextual noise in review-based recommendation via aspect term extraction and attention modeling. Information Sciences,**  [Paper Link](https://doi.org/10.1016/j.ins.2026.123078)

## Overview
This repository provides the official implementation of ATRS (Aspect Term-aware Recommender System), a review-based recommendation framework that enhances preference modeling by increasing the informational density of user reviews. ATRS addresses the limitation of existing methods that indiscriminately process entire reviews, where aspect-relevant content is often diluted by contextual noise. To mitigate this issue, ATRS employs a BERT-based aspect term extraction model to identify product-related terms and filter irrelevant information from reviews. The extracted aspect terms are then encoded using a convolutional neural network and aggregated through a self-attention mechanism to construct aspect-aware user and item representations. Experiments conducted on Amazon and Yelp datasets demonstrate that ATRS consistently outperforms representative baselines, achieving average improvements of 19.54% in MAE and 11.89% in RMSE, which confirms the effectiveness of aspect-level refinement and informational density optimization in review-based recommender systems.

## Requirements
- Python 3.10
- pandas==2.3.3
- numpy==2.2.6
- scikit-learn==1.7.2
- transformers==4.57.0
- torch==2.8.0
- torchvision==0.23.0
- pyarrow==21.0.0

## Repository Structure
Below is the project structure for quick reference.

```bash
├── data/                        # Dataset directory
│   ├── raw/                     # Original (unprocessed) datasets
│   └── processed/               # Preprocessed data for training/evaluation
│
├── model/                       # Model definitions and checkpoints
│
├── src/                         # Core source code
│   ├── data.py                  # data preprocessing module
│   ├── ate.py                   # Aspect term extraction module
│   ├── config.yaml              # Model and training configuration file
│   ├── path.py                  # Path and directory management utilities
│   └── utils.py                 # Helper functions (data loading, metrics, etc.)
│
├── main.py                      # Entry point for model training and evaluation
│
├── requirements.txt             # Python package dependencies
│
├── README.md                    # Project documentation
│
└── .gitignore                   # Git ignore configuration

```

## Model Description

ATRS (Aspect Term-aware Recommender System) is a review-based recommendation model designed to reduce contextual noise and enhance preference modeling by increasing the informational density of textual representations. Instead of processing entire reviews indiscriminately, ATRS explicitly focuses on aspect-level information that is directly relevant to target items.

The model consists of two main modules:
- ATE (Aspect Term Extraction): identifies product-related aspect terms from review text.
- RS (Recommender System): constructs aspect-aware user and item representations and predicts ratings.

In the ATE module, a Transformer-based encoder processes tokenized review text to capture contextual semantics. A Local Context Focus (LCF) mechanism refines token-level representations, and a BIO tagging scheme is applied to extract salient aspect terms.

In the RS module, the extracted aspect terms are embedded using a convolutional neural network and integrated with user and item latent vectors. A self-attention mechanism is employed to model the relative importance within each representation. The resulting user and item representations are then combined and passed to a rating prediction network, which models user–item interactions to predict ratings.

<p align="center">
  <img src="data/ATRS Architecture.png" alt="ATRS model Architecture" width="800">
</p>
