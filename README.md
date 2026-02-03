Named Entity Recognition (NER) with BERT Fine-Tuning

A complete end-to-end Named Entity Recognition (NER) system fine-tuned from a pretrained BERT (bert-base-cased) model on the WikiANN English dataset, 

built using Hugging Face Transformers & Datasets, and deployed with FastAPI + Streamlit + Docker (GPU-ready).

This project performs token-level classification to identify entities such as Persons, Organizations, and Locations, and exposes the model through a modern inference API and interactive web interface.

🚀 Key Features

Fine-tuned BERT (bert-base-cased) for English NER

Trained on WikiANN (PAN-X) English dataset with IOB2 tagging

Full training, evaluation, and inference pipeline

Metrics: Precision, Recall, F1-Score, Accuracy using seqeval

FastAPI backend for model inference

Streamlit web application for interactive usage

Dockerized & GPU-ready (CUDA 12.1, PyTorch 2.5.1)

Model published on Hugging Face Hub

Clean, modular, production-aware project structure

📂 Dataset Information

Dataset: WikiANN (PAN-X)

Source: Hugging Face Datasets

https://huggingface.co/datasets/unimelb-nlp/wikiann

Entity Labels:

PER — Person

ORG — Organization

LOC — Location

O — Outside (non-entity)

Tagging Format:

IOB2 (e.g., B-PER, I-PER, B-ORG, I-ORG)

Dataset Split (Re-balanced)

| Split      | Samples |

| ---------- | ------- |

| Train      | 30,000  |

| Validation | 5,000   |

| Test       | 5,000   |

🛠️ Tech Stack

Language: Python

Deep Learning: PyTorch, Hugging Face Transformers

Data Processing: Hugging Face Datasets, NumPy

Evaluation: seqeval, evaluate

API: FastAPI, Uvicorn

UI: Streamlit

Experiment Tracking: Weights & Biases (W&B)

Deployment: Docker, NVIDIA CUDA

📊 Model Performance (Test Set)

| Metric    | Score  |

| --------- | ------ |

| Accuracy  | ~93.0% |

| F1-Score  | ~84.3% |

| Precision | ~83.0% |

| Recall    | ~85.6% |

Training Configuration

Learning Rate: 2e-5

Epochs: 3

Optimizer: AdamW

Model: bert-base-cased

Metrics may vary slightly depending on random seed and hardware.

🤖 Model Access

7beshoyarnest/bert-finetuned-ner

https://huggingface.co/7beshoyarnest/bert-finetuned-ner

📦 Project Structure

Named_Entity_Recognition/

│

├── api/

│   ├── __init__.py

│   └── main.py              # FastAPI inference service

│

├── streamlit_app/

│   └── app.py               # Streamlit UI

│

├── Named_Entity_Recognition.ipynb              # Training & experiments

├── requirements.txt

├── Dockerfile

├── .dockerignore

└── README.md

⚙️ Installation & Usage

1️⃣ Clone the Repository

git clone https://github.com/7BeshoyArnest/Named_Entity_Recognition.git

cd Named_Entity_Recognition

2️⃣ Install Dependencies

pip install -r requirements.txt

🔍 Inference Examples

Option 1: Hugging Face Pipeline

from transformers import pipeline

ner = pipeline(
    "token-classification",
    model="7beshoyarnest/bert-finetuned-ner",
    aggregation_strategy="simple"
)

text = "The Burj Khalifa is located in Dubai and was built by Emaar Properties."

print(ner(text))

Option 2: FastAPI + Streamlit (Local)

# Start the API

uvicorn api.main:app --reload

# Run the Streamlit app

streamlit run streamlit_app/app.py

Open:

http://localhost:8502

🐳 Docker (GPU-Enabled)

Build Image

docker build -t ner-streamlit .

Run Container

docker run --gpus all -p 8502:8502 ner-streamlit

Requires NVIDIA Container Toolkit for GPU support.

🧠 Architecture Note

For portfolio and demo purposes, the FastAPI backend is launched inside the Streamlit process using a background thread.

In production environments, FastAPI should be deployed as a separate microservice.

🧑‍💻 Contributing

Contributions are welcome!

Fork the repository

Create a feature branch

Submit a pull request

🔮 Future Improvements

Separate FastAPI & Streamlit services (microservices)

Docker Compose deployment

Authentication & rate limiting

Batch inference & file upload

Multilingual / Arabic NER support

Model optimization & quantization









