Named Entity Recognition (NER) with BERT Fine‑Tuning

A Named Entity Recognition (NER) system fine‑tuned from a pretrained BERT model (bert‑base‑cased) on the WikiANN English dataset using the Hugging Face Transformers and Datasets libraries. This project performs 

token classification to identify and label entities like persons, locations, and organizations in text.

Dataset Source: [unimelb-nlp/wikiann on Hugging Face Datasets] 

🚀 Key Features

Fine‑tuned BERT (bert‑base‑cased) model for token‑level entity recognition.

Uses the WikiANN English dataset — a Wikipedia‑based NER corpus in IOB2 tagging format. 

Fully customizable training pipeline supporting evaluation metrics such as F1, precision, and recall.

Modular scripts for preprocessing, training, evaluation, saving, and inference.

Ready for integration with inference APIs or downstream NLP systems.

📂 Dataset Information

Dataset Link: https://huggingface.co/datasets/unimelb-nlp/wikiann

The WikiANN dataset classifies tokens into the following entity types:

LOC: Location

ORG: Organization

PER: Person

O: Outside (No entity)

The dataset is structured with IOB2 tagging (e.g., B-PER for the beginning of a person's name and I-PER for subsequent tokens).

Split,Samples

Train,"20,000"

Validation,"10,000"

Test,"10,000"

🛠️ Tech Stack

Language: Python

Deep Learning: PyTorch, Transformers (Hugging Face)

Data Processing: Datasets, NumPy

Evaluation: seqeval, evaluate

📊 Performance Results

Based on the fine-tuning process, the model achieves high accuracy across standard NER categories.

Metric,Score

Accuracy,~93.0%

F1-Score,~84.2%

Precision,~83.3%

Recall,~85.2%

Note: Metrics may vary slightly based on final hyperparameter tuning (Learning Rate: 2e-5, Epochs: 3, Batch Size: 32).

Installation & Usage

1-Clone the repository:

git clone https://github.com/7BeshoyArnest/Named_Entity_Recognition.git

cd Named_Entity_Recognition

2-Install requirements: 

pip install transformers datasets evaluate seqeval torch

pip install -r requirements.txt

3-Run Inference:

from transformers import pipeline

# Load your fine-tuned model

ner_model = pipeline("ner", model="bert-base-cased", aggregation_strategy="simple")

text = "The Burj Khalifa is located in Dubai and was built by Emaar Properties."

entities = ner_model(text)

print(entities)

OR Usin the Streamlit_app:

firstly: run the api using python -m uvicorn api.main:app --reload

secondly: run the streamlit app using python -m streamlit run streamlit_app/app.py

🧑‍💻 Contributing

Contributions and issues are welcome!

Please fork the repository and create pull requests for improvements.

