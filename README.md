# AI-Powered Retail CRM & Predictive Analytics Platform

An end-to-end machine learning platform built to analyze retail customer behavior and support decisions across customer retention, product recommendations, demand forecasting, inventory planning, and store layout.

## Overview

The project uses large-scale retail transaction data to build multiple predictive models and translate their outputs into actionable retail decisions.

The platform covers:

- Customer churn prediction
- Next-basket prediction
- Next-visit prediction
- Sales and inventory forecasting
- Product relationship analysis
- Store layout optimization
- Churn-prevention targeting

## Dataset

The project analyzes approximately **3.8M retail transactions across 40K customers**.

The dataset contains transaction-level information covering:

- Customer behavior
- Product and category information
- Transaction dates
- Sales value
- Quantity purchased
- Store zones

The underlying retail dataset is proprietary and is **not included in this repository**.

## Key Results

Performance varies slightly across model runs due to training initialization and execution conditions. The following represent the project's observed performance:

| Model | Objective | Representative Result |
|---|---|---:|
| Churn Prediction | Identify customers at risk of churn | **~93% ROC-AUC** |
| Next-Basket Prediction | Predict products likely to be purchased next | **~49% Recall@10** |
| Next-Visit Prediction | Predict customer's next visit | **~4-day MAE** |
| Inventory Forecasting | Forecast inventory requirements | **~84% accuracy** |

## Models

### 1. Customer Churn Prediction

Built a hybrid deep learning model combining customer purchase sequences with customer-level behavioral features.

The architecture uses:

- Bidirectional LSTM
- Transformer-based attention
- Word2Vec product embeddings
- Customer profile features

The model predicts whether a customer is likely to churn within a defined future window.

**Observed performance: ~93% ROC-AUC**

---

### 2. Next-Basket Prediction

Developed a sequence-based recommendation system to predict products a customer is likely to purchase in their next basket.

The model combines:

- Historical purchase sequences
- Product embeddings
- Customer-specific purchase behavior
- Transformer-based sequence modelling

**Observed performance: ~49% Recall@10**

---

### 3. Next-Visit Prediction

Built a customer-level forecasting model to estimate when a customer is likely to make their next visit.

Features incorporate historical purchasing behavior, customer characteristics, location-level patterns, and temporal signals.

**Observed performance: ~4-day MAE**

---

### 4. Sales & Inventory Forecasting

Developed forecasting models to support demand planning and inventory decisions across store zones.

The forecasting pipeline incorporates:

- Historical sales
- Calendar effects
- Seasonality
- Festival-period demand patterns
- Store-level information

**Observed inventory forecasting accuracy: ~84%**

---

### 5. Store Layout Optimization

Used product purchase relationships and product embeddings to identify products that are frequently associated with one another.

These relationships were used to generate:

- Product-to-zone assignments
- Store adjacency recommendations
- Product placement insights

The objective is to improve product discovery and leverage cross-purchase behavior.

---

### 6. Churn Prevention

Converted churn predictions into a targeted customer-retention workflow.

The approach identifies customers with elevated churn risk while considering recent customer activity to prioritize customers who are more actionable for retention campaigns.

## System Architecture

```text
                 Retail Transactions
                         │
                         ▼
              Data Cleaning & Feature
                   Engineering
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
   Customer Data    Purchase Data   Time-Series Data
          │              │              │
          ▼              ▼              ▼
     Churn Model   Recommendation   Forecasting
          │              │              │
          └──────────────┼──────────────┘
                         ▼
                Retail Decision Layer
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
    Retention       Merchandising    Inventory
    Targeting       & Placement       Planning



    Dataset: The project uses proprietary retail transaction data, which is not included in this repository.
