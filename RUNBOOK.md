# RUNBOOK.md

## Purpose

This runbook explains how to reproduce the main workflow of the **Hotel Analytics Decision Support System** end to end.

It is intended for a reasonably technical reader who wants to understand or reproduce:

- model development,
- artifact generation,
- batch scoring,
- AWS-based deployment setup,
- and the Streamlit demo application.

This document covers both:
1. **model / data workflow**, and
2. **demo / deployment workflow**.

---

## 1. Project Summary

This project includes two machine learning tasks:

1. **Booking Cancellation Prediction**
   - Input: booking-level hotel reservation data
   - Output: cancellation risk prediction

2. **Customer Value Classification**
   - Input: customer-level hotel history / profile data
   - Output: high-value vs. low-value classification

The final prototype exposes these tasks through a Streamlit application that includes:
- a project overview,
- a project dashboard,
- a strategy matrix,
- and a live prediction interface.

---

## 2. Required Components

To reproduce the project, you will need access to:

- the project repository
- Python 3.x
- project dependencies from `requirements.txt`
- the input datasets (if available)
- AWS resources or equivalent local substitutes, depending on how far you want to reproduce the pipeline

Optional but recommended:
- access to AWS S3
- access to SageMaker for notebook-based development
- access to EC2 for demo hosting
- access to RDS if reproducing the database layer

---

## 3. Repository Assumptions

This runbook assumes the repository contains folders similar to:

```text
.
├── README.md
├── RUNBOOK.md
├── requirements.txt
├── .gitignore
├── .env.example
│
├── src/
│   ├── train_cancellation.py
│   ├── train_customer_value.py
│   ├── score_cancellation.py
│   └── score_customer_value.py
│
├── live_app/
│   ├── app.py
│   ├── requirements.txt
│   ├── sample_data/
│   │   ├── sample_bookings.csv
│   │   └── sample_customers.csv
│   └── utils/
│       ├── cancellation_inference.py
│       ├── customer_value_inference.py
│       ├── recommendation.py
│       ├── db.py
│       └── s3_loader.py
│
├── notebooks/
│   ├── hotel_cancellation_prediction.ipynb
│   └── hotel_customer_value_classification.ipynb
│
├── artifacts/
│   ├── cancellation/
│   │   ├── eda_plots.png
│   │   ├── evaluation_plots.png
│   │   ├── feature_importance.png
│   │   ├── feature_list.pkl
│   │   ├── model_comparison.png
│   │   └── xgb_best_model.pkl
│   │
│   └── customer_value/
│       ├── customer_feature_list.pkl
│       ├── eda_customer_value.png
│       ├── evaluation_customer_value.png
│       ├── feature_importance_customer.png
│       ├── model_comparison_customer.png
│       └── xgb_customer_value_model.pkl
│
├── data/
│   ├── README.md
│   ├── raw/
│   │   ├── HotelCustomersDataset.xlsx
│   │   └── hotel_bookings.csv
│   ├── sample_inputs/
│   │   ├── sample_bookings.csv
│   │   └── sample_customers.csv
│   └── sample_outputs/
│       ├── cancellation_predictions.csv
│       └── customer_value_predictions.csv
│
└── docs/
    └── architecture_diagram.png
└── data/
    └── README.md
