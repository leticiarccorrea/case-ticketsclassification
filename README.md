# Ticket Classification – Data Science Case

## Overview

This project implements an end-to-end **text classification pipeline** to automatically categorize IT support tickets and assess their **criticality**. The solution was designed to handle **unstructured textual data**, class imbalance, and the absence of fully labeled historical data, which is a common real-world scenario in operational support systems.

The pipeline combines **weak supervision**, **classical NLP feature engineering**, and **robust linear models**, resulting in a production-ready and interpretable solution.

---

## Problem Statement

Given a dataset of IT support tickets containing free-text descriptions, the objectives are:

1. **Main Category Classification**
   Automatically assign each ticket to one of the following categories:
   - connectivity
   - Computer Hardware
   - Software & Operating System
   - Access, Permissions & Accounts  
   Tickets with low classification confidence are routed to an `other` fallback category.

2. **Criticality Classification**
   Independently classify each ticket as:
   - `urgent`
   - `normal`

Key challenges addressed:
- Lack of fully labeled historical data
- Highly imbalanced classes
- Informal language, typos, and heterogeneous writing styles
- Need for interpretability and safe fallback mechanisms

---

## Solution Approach

### Pipeline
1. Data ingestion
2. Text preparation and normalization
3. Pre-labeling (Weak Supervision) via Dictionary
4. Global Split (Single Hold-out)
5. Feature Engineering (TF-IDF Vectorization)
6. Model Training
7. Evaluation and Validation
8. Production Prediction (Unified Function)
9. Structured Output
{
  "main_category": "...",
  "criticality": "...",
  "category_margin": ...
}
  
---

## Results & Metrics (Summary)

### Category Classification
- **Test Accuracy:** ~0.76  
- **Test Macro F1:** ~0.61  
- Strong performance on dominant classes (e.g., Network)
- Expected confusion between semantically close categories
- Stable generalization confirmed via cross-validation

### Criticality Classification
- **Test Accuracy:** ~0.91  
- **Test Macro F1:** ~0.89  
- **Urgent F1:** ~0.84  
- High recall for urgent tickets, suitable for prioritization use cases

### Generalization & Robustness
- Train–test gap is expected for TF-IDF + linear models
- Cross-validation results closely match test performance
- No evidence of pathological overfitting

---

## Repository Structure

.
├── case_ticketclassification.ipynb
├── classificacao_atendimento.csv
└── README.md

---

## How to Run the Project Locally

### 1. Clone the Repository

git clone https://github.com/your-username/ticket-classification-case.git  
cd ticket-classification-case

---

### 2. Create a Virtual Environment (Recommended)

python -m venv venv  
source venv/bin/activate   # Linux / macOS  
venv\Scripts\activate      # Windows

---

### 3. Install Dependencies

pip install -r requirements.txt

Or manually:

pip install numpy pandas scikit-learn scipy jupyter

---

### 4. Run the Notebook

jupyter notebook

Open `case_ticketclassification.ipynb` and run all cells sequentially.

---

## Requirements

- Python 3.9+
- numpy
- pandas
- scikit-learn
- scipy
- jupyter

---

## Outputs

- Trained models for category and criticality
- Evaluation metrics and confusion matrices
- Feature importance analysis
- Unified prediction function returning:
  {
    "main_category": "...",
    "criticality": "...",
    "category_margin": ...
  }

---

## Notes

- The `other` category is a deliberate fallback for low-confidence predictions.
- The solution is designed for easy extension to active learning and semi-supervised approaches.
- Monitoring and retraining strategies are discussed in the notebook.

---

