# CV MLOps Pipeline

![CI/CD Pipeline](https://github.com/yourusername/cv-mlops-pipeline/actions/workflows/ci-cd.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.9-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.70%2B-009688?logo=fastapi)
![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-EE4C2C?logo=pytorch)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker)
![AWS Lambda](https://img.shields.io/badge/AWS-Lambda-FF9900?logo=amazonaws)
![License](https://img.shields.io/badge/license-MIT-green.svg)

A production-ready **Computer Vision MLOps pipeline** that exposes an object detection REST API built on **Faster R-CNN (ResNet-50 FPN)** trained on COCO. The service is containerized with Docker, deployable to AWS Lambda via Serverless Framework, and ships with a GitHub Actions CI/CD pipeline.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running Locally](#running-locally)
  - [Running with Docker](#running-with-docker)
- [API Reference](#api-reference)
- [Configuration](#configuration)
- [Testing](#testing)
- [Deployment](#deployment)
  - [Docker / DockerHub](#docker--dockerhub)
  - [AWS Lambda (Serverless)](#aws-lambda-serverless)
- [Monitoring & Metrics](#monitoring--metrics)
- [CI/CD](#cicd)

---

## Overview

This project demonstrates an end-to-end MLOps workflow for a computer vision inference service:

1. **Model** — Faster R-CNN with a ResNet-50 FPN backbone, capable of detecting **80 COCO object classes**.
2. **Serving** — FastAPI application with async request handling, lazy model initialization, and thread-safe singleton loading.
3. **Packaging** — Multi-stage Docker image based on `python:3.9-slim`.
4. **Cloud deployment** — AWS Lambda via [Mangum](https://github.com/jordaneremieff/mangum) ASGI adapter and Serverless Framework (`serverless.yml`).
5. **Observability** — CloudWatch metrics for API latency and prediction counts.
6. **Automation** — GitHub Actions pipeline that runs tests on every push/PR and auto-builds + pushes a Docker image on merges to `main`.

---

## Architecture

```
Client
  |
  v
FastAPI (uvicorn)  ──────────────────────────────┐
  |                                              |
  ├── GET  /            (root)                   |
  ├── GET  /health      (health check)           |  Local / Docker
  └── POST /predict     (image upload)           |
           |                                     |
           v                                     |
   ObjectDetectionModel                          |
   (Faster R-CNN ResNet-50 FPN)                  |
           |                                     |
           v                                    ─┘
   JSON response
   {predictions, model_version, processing_time}

AWS Lambda deployment path:
  API Gateway → Mangum → FastAPI → Model → CloudWatch Metrics
```

---

## Features

- **Object Detection** — 80-class COCO detection with bounding boxes and confidence scores (threshold > 0.5).
- **Lazy Model Loading** — Model is initialized on the first `/predict` call, keeping startup fast and tests lightweight.
- **Stub Mode** — Runs without PyTorch by default (`ENABLE_TORCH=0`), returning empty predictions. Flip `ENABLE_TORCH=1` to activate the real model.
- **Async API** — FastAPI with async file handling for high-throughput inference.
- **Health Endpoint** — `/health` for load-balancer and readiness probes.
- **CloudWatch Metrics** — Automatic latency and prediction count metrics when running on AWS.
- **Containerized** — Dockerfile ready for local development and CI/CD.
- **Serverless** — One-command deployment to AWS Lambda with API Gateway.

---

## Project Structure

```
cv-mlops-pipeline/
├── app/
│   ├── api/
│   │   └── endpoints.py        # FastAPI routes (/health, /predict)
│   ├── models/
│   │   └── model.py            # ObjectDetectionModel (Faster R-CNN wrapper)
│   ├── utils/
│   │   └── metrics.py          # CloudWatch MetricsCollector
│   ├── lambda_handler.py       # Mangum ASGI adapter for AWS Lambda
│   └── main.py                 # Uvicorn entrypoint
├── aws/
│   └── serverless.yml          # Serverless Framework configuration
├── tests/
│   └── test_api.py             # Pytest API tests
├── .github/
│   └── workflows/
│       └── ci-cd.yml           # GitHub Actions CI/CD pipeline
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.9+
- Docker (optional, for containerized runs)
- AWS CLI + Serverless Framework (optional, for cloud deployment)

### Installation

```bash
git clone https://github.com/yourusername/cv-mlops-pipeline.git
cd cv-mlops-pipeline

python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### Running Locally

Start the API server (stub mode, no PyTorch required):

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

To enable the real Faster R-CNN model:

```bash
ENABLE_TORCH=1 uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Running with Docker

```bash
docker build -t cv-mlops-pipeline .

# Stub mode
docker run -p 8000:8000 cv-mlops-pipeline

# Full model mode
docker run -p 8000:8000 -e ENABLE_TORCH=1 cv-mlops-pipeline
```

---

## API Reference

| Method | Endpoint   | Description                        |
|--------|------------|------------------------------------|
| GET    | `/`        | Welcome message                    |
| GET    | `/health`  | Health check — returns `{"status": "healthy"}` |
| POST   | `/predict` | Upload an image and get detections |

### POST `/predict`

**Request** — multipart/form-data

| Field | Type | Description        |
|-------|------|--------------------|
| file  | file | Image file (JPEG, PNG, etc.) |

**Response**

```json
{
  "predictions": [
    {
      "class": "person",
      "confidence": 0.987,
      "bbox": [x1, y1, x2, y2]
    }
  ],
  "model_version": "fasterrcnn_resnet50_fpn",
  "processing_time": 0.312,
  "api_latency": 0.345
}
```

**Example using the included client:**

```bash
python client.py
```

---

## Configuration

| Environment Variable | Default | Description                                  |
|----------------------|---------|----------------------------------------------|
| `ENABLE_TORCH`       | `0`     | Set to `1` to load the real PyTorch model    |
| `MODEL_PRETRAINED`   | `0`     | Set to `1` to use COCO pretrained weights    |
| `STAGE`              | `dev`   | Deployment stage (used in Lambda config)     |

---

## Testing

```bash
pytest
```

Tests cover:
- `GET /` — root endpoint response
- `GET /health` — health check response

---

## Deployment

### Docker / DockerHub

The CI/CD pipeline automatically builds and pushes to DockerHub on merges to `main`. To push manually:

```bash
docker build -t yourusername/cv-mlops-pipeline:latest .
docker push yourusername/cv-mlops-pipeline:latest
```

### AWS Lambda (Serverless)

```bash
npm install -g serverless

# Deploy to dev stage
serverless deploy --stage dev

# Deploy to production
serverless deploy --stage prod
```

The `serverless.yml` configures:
- **Runtime**: Python 3.9
- **Region**: us-east-1
- **Memory**: 1024 MB
- **Timeout**: 30 seconds
- **Routing**: API Gateway proxy (`/{proxy+}`)

---

## Monitoring & Metrics

When deployed on AWS, the `MetricsCollector` class publishes the following custom CloudWatch metrics under the `CVMLOpsMetrics` namespace:

| Metric             | Unit         | Dimension        | Description                          |
|--------------------|--------------|------------------|--------------------------------------|
| `APILatency`       | Milliseconds | `Endpoint`       | End-to-end latency per API endpoint  |
| `PredictionCount`  | Count        | `Type`           | Number of prediction requests        |

---

## CI/CD

The GitHub Actions workflow (`.github/workflows/ci-cd.yml`) runs on every push and pull request to `main`:

1. **Test job** — sets up Python 3.9, installs dependencies, runs `pytest`.
2. **Build & Push job** — runs only on `push` to `main` after tests pass; builds the Docker image and pushes to DockerHub using `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN` repository secrets.
