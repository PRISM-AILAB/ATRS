# ATRS

Official implementation of:
> Lim, H., Li, X., Park, S., Li, Q., & Kim, J. (2026). 
**Reducing contextual noise in review-based recommendation via aspect term extraction and attention modeling**. 
_Information Sciences_, 735, 123078.  [Paper](https://doi.org/10.1016/j.ins.2026.123078)

## Overview
This repository provides the official implementation of ATRS (Aspect Term-aware Recommender System), a review-based recommendation model that enhances preference modeling by increasing the informational density of user reviews. ATRS addresses the limitation of existing methods that indiscriminately process entire reviews, where aspect-relevant content is often diluted by contextual noise. To mitigate this issue, ATRS employs a BERT-based aspect term extraction (ATE) module to identify product-related terms and filter irrelevant information from reviews. The extracted aspect terms are then encoded using a convolutional neural network and aggregated through a self-attention mechanism to construct aspect-aware user and item representations. Experiments conducted on Amazon and Yelp datasets demonstrate that ATRS consistently outperforms representative baselines, achieving average improvements of 19.54% in MAE and 11.89% in RMSE. These results highlight the effectiveness of aspect-level refinement in review-based recommender systems.

## Requirements
- ﻿python>=3.9
- ﻿torch>=2.5.1
- torchvision>=0.20.1
- numpy==1.26.4
- pandas>=2.0.0
- gensim==4.3.3
- pyarrow>=12.0.0
- scikit-learn>=1.5.0
- tqdm>=4.66.0
- PyYAML>=6.0.0
- sentencepiece==0.2.1
- transformers==4.29.2
- tokenizers==0.13.3
- pyabsa==2.4.3
- nltk>=3.9.0
- seqeval>=1.2.0
- termcolor>=2.0.0

## Repository Structure
Below is the project structure for quick reference.

```bash
├── data/                        # Dataset directory
│   ├── raw/                     # Original (unprocessed) datasets
│   └── processed/               # Preprocessed data for training and evaluation
│
├── model/                       # Model definitions and checkpoints
│   └── proposed.py              # ATRS model architecture and training utilities
│
├── src/                         # Core source code
│   ├── data.py                  # Data preprocessing module
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

ATRS (Aspect Term-aware Recommender System) is a review-based recommendation model designed to reduce contextual noise and enhance preference modeling through aspect-level refinement of textual representations. Instead of processing entire reviews indiscriminately, ATRS explicitly focuses on item-relevant aspect information.

The model consists of two main modules:
- **ATE (Aspect Term Extraction) Module:** Identifies product-related aspect terms from review text.
- **RS (Recommender System) Module:** Constructs aspect-aware user and item representations and predicts ratings.

In the ATE module, a Transformer-based encoder processes tokenized review text to capture contextual semantics. A Local Context Focus (LCF) mechanism further refines token-level representations, and a BIO tagging scheme is applied to extract salient aspect terms.

In the RS module, the extracted aspect terms are embedded using a convolutional neural network and integrated with user and item latent embeddings. A self-attention mechanism models the relative importance within each representation. The refined user and item representations are then combined and passed to a rating prediction network for final score estimation.

<p align="center">
  <img src="data/ATRS Architecture.png" alt="ATRS model Architecture" width="800">
</p>

## How to Run

### Environment Setup
Create a virtual environment (Python ≥ 3.9 recommended) and install the required dependencies:

#### Option A: Using venv
```bash
python3.9 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

#### Option B: Using Conda
```bash
conda create -n atrs python=3.9
conda activate atrs
pip install -r requirements.txt
```

### Data Preparation
Place your dataset under `data/raw/` and ensure that its format matches the preprocessing pipeline defined in `src/data.py`.

### Configuration
Edit `src/config.yaml` to set training, data, and model hyperparameters before running.

### Train and Evaluate the Model
Run the training script:
```bash
python main.py
```

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
        <td><b>Proposed (ATRS)</b></td>
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

_Last updated: March 2026_
