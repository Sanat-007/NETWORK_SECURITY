# Network Security ML Pipeline

An end-to-end Machine Learning project that implements a modular training and prediction pipeline for network security data. The project covers the complete ML lifecycle, including data ingestion, validation, transformation, model training, experiment tracking, and model serving through a FastAPI application.

The primary goal of this project is to apply software engineering and MLOps principles to a machine learning workflow while maintaining a clean and scalable project structure.

---

## Overview

This project is structured as a production-inspired machine learning pipeline where each stage of the workflow is isolated into independent components.

The pipeline:

1. Ingests data from MongoDB
2. Validates incoming datasets against predefined schemas
3. Performs data preprocessing and feature transformation
4. Trains and evaluates machine learning models
5. Tracks experiments using MLflow
6. Stores model artifacts for future inference
7. Serves predictions through a FastAPI application

---

## Key Features

- Modular pipeline architecture
- MongoDB-based data ingestion
- Automated data validation
- Data preprocessing and transformation
- Model training and evaluation
- Experiment tracking with MLflow
- Model artifact management
- FastAPI-powered prediction service
- Docker support for containerized deployment
- Structured logging and exception handling

---

## Project Architecture

```text
                 ┌─────────────┐
                 │  MongoDB    │
                 └──────┬──────┘
                        │
                        ▼
             ┌──────────────────┐
             │ Data Ingestion   │
             └────────┬─────────┘
                      │
                      ▼
             ┌──────────────────┐
             │ Data Validation  │
             └────────┬─────────┘
                      │
                      ▼
             ┌──────────────────────┐
             │ Data Transformation  │
             └────────┬─────────────┘
                      │
                      ▼
             ┌──────────────────┐
             │ Model Training   │
             └────────┬─────────┘
                      │
                      ▼
             ┌──────────────────┐
             │ Model Evaluation │
             └────────┬─────────┘
                      │
                      ▼
             ┌──────────────────┐
             │ MLflow Tracking  │
             └────────┬─────────┘
                      │
                      ▼
             ┌──────────────────┐
             │ Saved Artifacts  │
             └────────┬─────────┘
                      │
                      ▼
             ┌──────────────────┐
             │ FastAPI Service  │
             └──────────────────┘
```

---

## Tech Stack

| Category | Technologies |
|-----------|-------------|
| Programming Language | Python |
| Machine Learning | Scikit-Learn |
| Data Processing | Pandas, NumPy |
| API Framework | FastAPI |
| Database | MongoDB |
| Experiment Tracking | MLflow |
| Deployment | Docker |
| Version Control | Git & GitHub |

---

## Repository Structure

```bash
NETWORK_SECURITY/
│
├── networksecurity/
│   │
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_validation.py
│   │   ├── data_transformation.py
│   │   ├── model_trainer.py
│   │   └── model_evaluation.py
│   │
│   ├── pipeline/
│   │   ├── training_pipeline.py
│   │   └── prediction_pipeline.py
│   │
│   ├── entity/
│   ├── constant/
│   ├── exception/
│   ├── logging/
│   └── utils/
│
├── data_schema/
├── final_models/
├── templates/
│
├── app.py
├── push_data.py
├── Dockerfile
├── requirements.txt
├── setup.py
└── README.md
```

---

## Machine Learning Workflow

### Data Ingestion

The ingestion component retrieves data from MongoDB and prepares it for downstream processing.

### Data Validation

The validation stage ensures:

- Schema consistency
- Required feature availability
- Dataset integrity checks

### Data Transformation

The transformation pipeline:

- Cleans input data
- Applies preprocessing
- Generates transformed datasets
- Saves preprocessing artifacts

### Model Training

The training component:

- Trains machine learning models
- Compares model performance
- Selects the best-performing model

### Model Evaluation

The evaluation stage measures model performance on unseen data and logs metrics for comparison.

### Model Serving

The trained model is exposed through a FastAPI application that supports batch prediction using CSV files.

---

## Installation

### Clone the Repository

```bash
git clone https://github.com/Sanat-007/NETWORK_SECURITY.git
cd NETWORK_SECURITY
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

**Windows**

```bash
venv\Scripts\activate
```

**Linux / macOS**

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Environment Configuration

Create a `.env` file in the project root directory.

```env
MONGO_DB_URL=<your_mongodb_connection_string>
```

Example:

```env
MONGO_DB_URL=mongodb+srv://username:password@cluster.mongodb.net/
```

---

## Running the Application

Start the FastAPI server:

```bash
uvicorn app:app
```

Application URL:

```text
http://localhost:8000
```

Swagger Documentation:

```text
http://localhost:8000/docs
```

---

## API Endpoints

### Train Model

```http
GET /train
```

Runs the complete machine learning pipeline:

- Data Ingestion
- Data Validation
- Data Transformation
- Model Training
- Model Evaluation

### Predict

```http
POST /predict
```

Upload a CSV file to generate predictions using the trained model.

---

## MLflow Integration

This project uses MLflow to track and manage machine learning experiments.

Tracked information includes:

- Model parameters
- Evaluation metrics
- Trained model artifacts
- Experiment history

Run MLflow UI:

```bash
mlflow ui
```

---

## Docker Support

### Build Docker Image

```bash
docker build -t network-security .
```

### Run Docker Container

```bash
docker run -p 8000:8000 network-security
```

---

## Engineering Practices

- Modular project structure
- Separation of concerns
- Configuration-driven workflow
- Artifact-based pipeline communication
- Centralized logging
- Custom exception handling
- Experiment tracking
- Reproducible training workflow

---

## Future Improvements

- CI/CD Pipeline Integration
- Model Registry Support
- Cloud Deployment
- Automated Retraining
- Monitoring Dashboard
- Feature Store Integration

---

## Learning Outcomes

Through this project, I gained practical experience with:

- End-to-end Machine Learning pipelines
- MLOps workflows
- MLflow experiment tracking
- FastAPI development
- MongoDB integration
- Model deployment concepts
- Software engineering practices for ML systems

---

## Author

**Sanat**

GitHub: https://github.com/Sanat-007

---

⭐ If you found this project useful, consider giving it a star.