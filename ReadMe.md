# Customer Churn Prediction & Explanations Dashboard

![Project Demo Screenshot](<PATH_TO_YOUR_SCREENSHOT.png>)

## 🚀 Project Overview

This project is an end-to-end data science application designed to predict customer churn for a fictional telecommunications company. It features an interactive web dashboard built with Plotly Dash that not only provides churn predictions but also uses SHAP (SHapley Additive exPlanations) to explain the key factors driving each prediction.

The primary goal is to empower business stakeholders (e.g., marketing, customer retention teams) to make data-driven decisions by understanding *why* a customer is at risk, allowing for targeted intervention strategies.

## ✨ Key Features

- **End-to-End ML Pipeline:** Includes data preprocessing, feature engineering, and model training.
- **High-Performance Modeling:** Utilizes an XGBoost classifier, optimized to handle class imbalance.
- **Model Explainability:** Integrates SHAP to generate easy-to-understand waterfall plots, showing the impact of each feature on the final prediction.
- **Interactive Web Dashboard:** A user-friendly interface where users can input a `customerID` and get an instant, explainable prediction.
- **Containerized Application:** Fully containerized with Docker for easy, consistent deployment on any machine.

## 🛠️ Tech Stack

- **Data Science & ML:** Python, Pandas, Scikit-learn, XGBoost, SHAP
- **Web Dashboard:** Plotly Dash, Gunicorn
- **Deployment:** Docker

## 🔧 Getting Started

To get a local copy up and running, follow these simple steps.

### Prerequisites

You must have **Docker** installed on your machine.
- [Install Docker](https://www.docker.com/products/docker-desktop/)

### Installation & Run

1. **Clone the repository:**
   ```sh
   git clone [https://github.com/bindugayathri/customer-churn-dashboard.git]
