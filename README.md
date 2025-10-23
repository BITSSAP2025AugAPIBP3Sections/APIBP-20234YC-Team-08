# Handwritten Character Recognition ML

# Local Setup

1. `git clone git@github.com:BITSSAP2025AugAPIBP3Sections/APIBP-20234YC-Team-08.git`

2. `python -m venv`

3. `.\venv\Scripts\activate.cmd`

4. `pip install -r  requirements.txt`

5. `streamlit run app.py`

Once you are done, do `deactivate`

# Introduction

This project is a machine learning application developed using Python for the backend, with GitHub Actions managing automated pipelines and deployment hosted on Microsoft Azure. The system leverages the MNIST dataset to train a model capable of recognizing handwritten digits with high accuracy.

# Aim

The primary aim of this project is to build an end-to-end digit recognition system that demonstrates the integration of machine learning models into a scalable cloud-based application with automated workflows.

# Techstack

* Frontend: Python (temporarily) 
* Backend: Python
* Cloud provider: Azure
* Pipeline: GitHub Actions

# Applications

* Automatic digit recognition in forms and documents (e.g., bank cheques, post codes, surveys).

* Educational tools for teaching machine learning concepts.

* Integration into digital data entry systems to reduce manual input errors.

* Prototype for more complex handwriting recognition or OCR systems.

# APIs

## Core Recognition Endpoints

### 1. Single Image Recognition
```
POST /api/v1/predict
Content-Type: multipart/form-data or application/json
```
- Upload single image for digit recognition
- Returns predicted digit with confidence score

### 2. Batch Image Recognition
```
POST /api/v1/predict/batch
Content-Type: multipart/form-data
```
- Process multiple images simultaneously
- Returns array of predictions with confidence scores

## Model Management Endpoints

### 3. Model Information
```
GET /api/v1/model/info
```
- Returns current model version, accuracy metrics, training date

### 4. Model Health Check
```
GET /api/v1/health
```
- Check if the model service is running properly
- Returns status and response time

## Advanced Features

### 5. Confidence Threshold Prediction
```
POST /api/v1/predict/threshold
Content-Type: application/json
Body: {"image": "base64_data", "threshold": 0.8}
```
- Only return predictions above specified confidence level

## Utility Endpoints

### 6. Supported Formats
```
GET /api/v1/formats
```
- List supported image formats and size limits

# Team members:

* Pranav. P. A (2023sl70045)
* Varun. Venkatesh (2023sl70038)
* Anand Sai (2023sl70035)
* Shreesha Hegde (2023sl70018)

