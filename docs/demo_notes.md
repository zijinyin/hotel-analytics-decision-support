# Demo Notes

## Purpose

This document summarizes how to present and demonstrate the **Hotel Analytics Decision Support System** in a clear and structured way.

The demo is designed to show:
- the business motivation behind the project,
- the final model choices,
- the AWS deployment story,
- and the live user experience of the system.

---

## App Structure

The Streamlit demo application currently contains four pages:

1. **Overview**
2. **Project Dashboard**
3. **Strategy Matrix**
4. **Live Prediction**

Each page serves a different role in the presentation.

---

## 1. Overview

### Purpose
The Overview page introduces the problem and explains what the system is trying to do.

### Key points to mention
- Hotels need to identify which bookings are more likely to be canceled.
- Hotels also want to understand which customers are more valuable.
- The system combines both signals to support better operational decision-making.
- The intended end users are booking operations teams, revenue managers, and guest service teams.

### Suggested explanation
Use this page to frame the project briefly before going into technical details.

---

## 2. Project Dashboard

### Purpose
The Project Dashboard summarizes the modeling work behind the prototype.

### What this page shows
- dataset sizes
- preprocessing summary
- feature engineering highlights
- model comparison
- best-performing model shown in dashboard
- AWS deployment summary

### Key points to mention
- Two tasks were modeled:
  - booking cancellation prediction
  - customer value classification
- Multiple models were compared for both tasks.
- Random Forest is shown in the dashboard as the strongest-performing model in offline evaluation.
- The live deployed prototype uses the packaged XGBoost pipeline that was integrated into the demo workflow.

### Suggested explanation
This page helps transition from “what the problem is” to “how we solved it.”

---

## 3. Strategy Matrix

### Purpose
The Strategy Matrix page translates model outputs into practical hotel actions.

### What this page shows
- a 2×2 matrix:
  - booking risk on one axis
  - customer value on the other
- four simulated personas / cases
- hover-based summary on the chart
- a detailed recommendation section for a selected case

### Four strategy zones
- **Q1: High Risk + High Value**
  - highest priority
  - immediate follow-up recommended
- **Q2: Low Risk + High Value**
  - maintain relationship
  - strong service attention, but no urgent intervention
- **Q3: High Risk + Low Value**
  - monitor efficiently
  - lower-cost operational handling
- **Q4: Low Risk + Low Value**
  - normal workflow
  - no special intervention needed

### Suggested explanation
This page is useful for explaining how booking risk and customer value can be combined into strategy, even if the live demo itself focuses mainly on the booking side.

---

## 4. Live Prediction

### Purpose
This is the main live-demo page and should be the centerpiece of the presentation.

### What this page allows the user to do
- enter booking information
- select a sample customer profile
- generate:
  - booking cancellation prediction
  - customer value prediction
  - recommendation text

### Recommended demo flow
For the live presentation, focus first on the booking cancellation side.

Suggested structure:
1. Introduce the interface
2. Show a lower-risk booking case
3. Modify a few booking features
4. Show a higher-risk booking case
5. Explain how the output changes
6. Briefly mention that the full system can also combine booking and customer information together

### Good booking features to change during the live demo
To demonstrate higher or lower cancellation risk, useful features include:
- `lead_time`
- `previous_cancellations`
- `deposit_type`
- `is_repeated_guest`
- `previous_bookings_not_canceled`

### Example demo logic
#### Lower-risk case
Use:
- shorter or moderate lead time
- no previous cancellations
- more committed deposit type
- stronger booking history

Expected interpretation:
- lower booking risk
- more stable reservation

#### Higher-risk case
Change:
- higher lead time
- more previous cancellations
- no deposit
- lower repeat or weaker booking history

Expected interpretation:
- higher booking risk
- less stable reservation

### Important note
Even if the live spoken demo focuses mainly on booking risk, the system architecture still supports both:
- booking prediction
- customer value prediction

At the end of the live demo, mention that a real hotel workflow could use both signals together.

---

## Recommended Presentation Flow

A good 8–10 minute flow is:

### 1. Business Problem & End User
Briefly explain:
- what the problem is,
- why it matters,
- and who the system is for.

### 2. Data & Final Model
Explain:
- the two datasets,
- the final model comparison,
- and why the deployed prototype uses the current XGBoost pipeline.

### 3. Cloud Architecture
Explain:
- S3
- SageMaker
- RDS
- EC2
- Streamlit

### 4. Live Demo
Focus on:
- the app structure
- the Live Prediction page
- a lower-risk example
- a higher-risk example

### 5. Lessons Learned & Next Steps
Explain:
- what worked,
- what was difficult,
- and how the system could evolve into a daily-refresh batch pipeline.

---

## Recommended Talking Points for the Live Demo

### Intro
“This page represents the user-facing prediction interface. A hotel user can enter booking information, select a customer profile, and receive prediction outputs immediately.”

### Lower-risk case
“For this example, I’ll keep the booking in a more stable configuration, such as lower lead time and no cancellation history, to show how the system interprets a lower-risk booking.”

### Higher-risk case
“Now I’ll increase some risk-related features, such as lead time and prior cancellation history, and reduce booking commitment through deposit choice, to show how the system responds to a riskier scenario.”

### Closing line
“This demonstrates how the interface can support real-time booking evaluation. In a fuller operational setting, hotel teams could use both the booking signal and the customer signal together to prioritize action.”

---

## Demo Readiness Checklist

Before presenting, verify the following:

- [ ] EC2 instance is running
- [ ] current EC2 public IP is correct
- [ ] Streamlit app is already launched
- [ ] browser can access the app
- [ ] all tabs load correctly
- [ ] model artifacts are present
- [ ] sample customer file is present
- [ ] the selected demo cases have been tested in advance
- [ ] backup screenshots are available

---

## Optional Future Demo Improvements

Possible future improvements to the demo include:
- loading daily batches of bookings automatically
- surfacing multiple risky bookings at once
- connecting customer and booking records more directly
- showing a refreshed operational dashboard each day
- highlighting which bookings and customers deserve the most urgent attention
