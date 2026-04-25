# Hotel Analytics Decision Support System

## Overview

This project builds a hotel decision-support prototype that combines two machine learning tasks:

1. **Booking Cancellation Prediction**  
   Predict whether an individual hotel booking is likely to be canceled.

2. **Customer Value Classification**  
   Classify whether a customer is high value or low value.

The overall goal is to support hotel operations and revenue-management decisions by helping users answer two practical questions:

- **Which bookings are most at risk of cancellation?**
- **Which customers are most important to prioritize?**

Instead of treating all bookings and all customers the same way, this system is designed to help hotel teams allocate attention and follow-up more intelligently.

The final prototype includes:
- data cleaning and feature engineering workflows,
- model training and evaluation code,
- batch prediction workflows,
- AWS deployment components,
- and a Streamlit-based live demo application.

---

## Business Problem

Hotels regularly face uncertainty from booking cancellations. When cancellations happen late or unexpectedly, they can create problems for:

- occupancy planning,
- staffing,
- revenue forecasting,
- and customer experience management.

At the same time, not all guests are equally important from a business perspective. Some customers may have higher revenue potential, stronger loyalty, or greater long-term strategic value.

This creates a real operational problem:

> A hotel may want to intervene differently depending on both the **risk of losing the booking** and the **value of the customer**.

This project addresses that problem by combining:
- a **booking-level risk model**, and
- a **customer-level value model**

into a single decision-support prototype.

A “win” for the end user means being able to:
- identify risky bookings earlier,
- recognize which customers deserve more attention,
- and translate predictions into more practical operational action.

---

## End Users

This system is intended for users such as:

- hotel revenue managers,
- booking operations teams,
- guest experience / service teams,
- and customer relationship or retention teams.

The prototype is not meant to replace human decision-making. Instead, it is designed to provide a clearer signal about where attention may be most valuable.

---

## Project Scope

This project includes two main layers:

### 1. Model Development Layer
This includes:
- data cleaning,
- feature engineering,
- model training,
- validation,
- and performance comparison.

### 2. Demo / Decision-Support Layer
This includes:
- project overview pages,
- model and dataset summary pages,
- a strategy matrix for interpreting booking risk and customer value together,
- and a live prediction interface.

---

## Datasets

## Dataset 1 — Booking Cancellation

This dataset is used for **booking-level cancellation prediction**.

### Original size
- **119,390 rows**
- **32 columns**

### Modeling task
Predict whether a booking will be canceled.

### Example preprocessing steps
- removed invalid rows where total guests = 0,
- excluded leakage-prone variables,
- handled missingness in fields such as `children`, `agent`, and `company`,
- performed feature engineering on booking behavior and booking structure.

### Final processed design
- Final modeled matrix: **119,210 × 34**

### Example engineered features
- `total_nights`
- `total_guests`
- `has_agent`
- `has_company`
- `has_special_requests`
- `lead_time_log`
- `adr_log`
- `agent_log`
- `company_log`
- `cancel_rate_history`

---

## Dataset 2 — Customer Value

This dataset is used for **customer-level value classification**.

### Original size
- **83,590 rows**

### Filtered modeling subset
- **63,670 active customers**

### Modeling task
Classify whether a customer is high value or low value.

### Example preprocessing steps
- filtered to customers with at least one checked-in booking,
- removed anonymized ID/hash fields from modeling,
- cleaned and imputed age-related values,
- created target labels using median lodging revenue.

### Final processed design
- Final modeled matrix: **63,670 × 31**

### Example engineered features
- `total_special_requests`
- `has_canceled`
- `has_noshowed`
- `is_repeat`
- `tenure_days`
- `AverageLeadTime_log`
- `DaysSinceCreation_log`

---

## Modeling Summary

## Booking Cancellation Task

The following models were compared:
- Majority class baseline
- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost (paper replica)
- Tuned XGBoost

### Held-out test performance
- **Random Forest**:  
  - F1 = **0.8126**  
  - AUC = **0.9338**

- **Tuned XGBoost**:  
  - F1 = **0.7887**  
  - AUC = **0.9255**

### Additional note
The Random Forest model produced the strongest offline evaluation performance in the final comparison for this task.

---

## Customer Value Task

The following models were compared:
- Majority class baseline
- Logistic Regression
- Random Forest
- XGBoost

### Held-out test performance
- **Random Forest**:  
  - F1 = **0.7704**  
  - AUC = **0.8549**

- **XGBoost**:  
  - F1 = **0.7540**  
  - AUC = **0.8223**

### Additional note
The Random Forest model also produced the strongest offline evaluation performance in this task.

---

## Final Model Interpretation

There are two slightly different “final model” views in this project:

### For model reporting
Random Forest is shown as the strongest-performing model on both tasks in the project dashboard, because it achieved the best held-out performance among the compared models.

### For the deployed live prototype
The live AWS demo uses the XGBoost artifacts that were packaged and tested as part of the deployed inference workflow.

This means:
- **Random Forest** is highlighted in the reporting layer as the strongest evaluated model.
- **XGBoost** is used in the deployed live app because it was already integrated into the end-to-end AWS prototype pipeline.

---

## Streamlit Demo Application

The final user-facing prototype is a Streamlit application with four main pages:

### 1. Overview
Provides a short description of:
- the business problem,
- the modeling tasks,
- and the high-level purpose of the system.

### 2. Project Dashboard
Summarizes:
- dataset sizes,
- preprocessing highlights,
- feature engineering,
- model comparison results,
- and AWS system status.

### 3. Strategy Matrix
Shows a simulated 2×2 decision matrix that combines:
- booking risk
- customer value

This page is used to illustrate how operational strategies can differ depending on whether a case is:
- high-risk / high-value,
- low-risk / high-value,
- high-risk / low-value,
- or low-risk / low-value.

### 4. Live Prediction
Allows a user to:
- enter booking features,
- select a sample customer profile,
- generate a booking cancellation prediction,
- generate a customer value prediction,
- and view a simple recommendation.

---

## AWS Architecture

The deployed prototype uses the following AWS components:

### Amazon S3
Used for:
- raw dataset storage,
- notebook and script storage,
- model artifacts,
- and prediction output files.

### Amazon SageMaker
Used for:
- notebook-based experimentation,
- model development,
- training,
- evaluation,
- and artifact generation.

### Amazon RDS MySQL
Used as a structured database layer for:
- prediction outputs,
- metadata,
- and future system extensions.

### Amazon EC2
Used to host the Streamlit application that serves as the final demo interface.

---

## End-to-End Workflow

A high-level project workflow is:

1. Prepare and clean the booking and customer datasets
2. Engineer features for each task
3. Train and compare multiple models
4. Save model artifacts and feature lists
5. Run batch scoring
6. Store or transfer prediction outputs
7. Launch the Streamlit application in AWS
8. Use the UI for reporting and live decision support

---

## Repository Structure

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
│   ├── score_customer_value.py
│   └── load_to_rds.py                # optional if included
│
├── live_app/
│   ├── app.py
│   ├── sample_data/
│   │   ├── sample_bookings.csv
│   │   └── sample_customers.csv
│   └── utils/
│       ├── cancellation_inference.py
│       ├── customer_value_inference.py
│       └── recommendation.py
│
├── notebooks/
│   ├── hotel_cancellation_prediction.ipynb
│   └── hotel_customer_value_classification.ipynb
│
├── artifacts/
│   ├── feature_list.pkl
│   ├── customer_feature_list.pkl
│   ├── xgb_best_model.pkl
│   └── xgb_customer_value_model.pkl
│
├── docs/
│   ├── architecture_diagram.png
│   └── demo_notes.md
│
└── data/
    └── README.md