# AI Detector

An NLP-based machine learning project that classifies text as either human-written or AI-generated using TF-IDF feature extraction and linear classification.

## Overview

This project uses **Term Frequency-Inverse Document Frequency (TF-IDF)** to extract key linguistic features from written text and trains a linear classifier (`PassiveAggressiveClassifier` / `SGDClassifier`) to detect patterns indicative of generated text versus human authorship.

## Dataset

The model trains on the `AI_Human.csv` dataset.

- Due to GitHub file size limits, the raw dataset is not tracked in this repository.
- Download `AI_Human.csv` and place it in the project root directory before training.

## Getting Started

### Prerequisites

Install the required dependencies:

```bash
pip install numpy pandas scikit-learn joblib
```
