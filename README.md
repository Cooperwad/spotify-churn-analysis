# Spotify User Clustering for Churn Analysis

This repository is for our CISC483 final project focused on clustering Spotify users into interpretable listener segments and examining how those segments relate to churn behavior.


## Project Overview

Using the **Spotify Analysis Dataset 2025** from Kaggle, this project will explore whether Spotify users can be grouped into meaningful listener segments based on demographic, subscription, device, and listening-behavior features.

The central research question is:

> Can Spotify users be clustered into interpretable listener groups, and do those groups reveal meaningful differences in churn behavior?

## Planned Approach

The project is currently scaffolded for the following workflow:

1. Explore and clean the dataset.
2. Encode categorical features and scale numeric features where needed.
3. Perform exploratory data analysis on listening behavior and churn patterns.
4. Build a baseline clustering approach with K-means.
5. Compare alternative clustering methods such as hierarchical clustering, Gaussian mixture models, or DBSCAN.
6. Evaluate cluster quality and compare churn rates across groups.
7. Optionally test whether cluster membership improves downstream churn classification.

## Repository Structure

```text
.
├── data/
│   ├── raw/           # Original Kaggle dataset files
│   └── processed/     # Cleaned, transformed, or feature-engineered data
├── docs/              # Project notes and supporting documentation
├── models/            # Saved clustering artifacts or downstream models
├── notebooks/         # EDA, preprocessing, clustering, and evaluation notebooks
├── references/        # Proposal, data dictionary, and external references
├── reports/
│   └── figures/       # Plots, charts, and visuals for presentation/reporting
└── src/               # Reserved for reusable project code later
```

## Data Notes

- Put the original Kaggle download in `data/raw/`.
- Store cleaned or modeled datasets in `data/processed/`.
- Large data files and generated artifacts are ignored by git in this initial setup.

## Evaluation Plan

Planned evaluation includes:

- Clustering metrics such as silhouette score and Davies-Bouldin index
- Visual inspection of cluster separation
- Churn-rate comparisons across clusters
- Classification metrics such as F1 score, ROC-AUC, and precision-recall if a supervised extension is added

## Setup
- Install requirements from `requirements.txt` with `pip install -r requirements.txt`. 
- Add the data from Kaggle to `data/raw/`.
- Run `python src/clean_data.py` from the project root to generate processed data. 

