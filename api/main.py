from fastapi import FastAPI, HTTPException, status
from transformers import AutoTokenizer, AutoModelForTokenClassification
from pydantic import BaseModel
import torch

app = FastAPI()

class InputText(BaseModel):
    text: str

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("7beshoyarnest/bert-finetuned-ner")
model = AutoModelForTokenClassification.from_pretrained("7beshoyarnest/bert-finetuned-ner")
model.eval()

# Get label names from the model config
label_names = model.config.id2label

@app.post("/predict")
def predict_ner(input: InputText):
    if not input.text or input.text.strip() == "":
        raise HTTPException(
            status_code=400, 
            detail="No input provided. Please enter some text."
        )
    
    # Tokenize input
    encoded_input = tokenizer(
        input.text, 
        return_tensors="pt", 
        truncation=True, 
        padding=True,
        is_split_into_words=False
    )
    
    with torch.no_grad():
        outputs = model(**encoded_input)
        logits = outputs.logits  # shape: (batch_size, seq_len, num_labels)
        predictions = torch.argmax(logits, dim=-1)  # predicted labels per token

    tokens = tokenizer.convert_ids_to_tokens(encoded_input["input_ids"][0])
    preds = predictions[0].tolist()

    # Build NER result
    entities = []
    for token, pred_id in zip(tokens, preds):
        label = label_names[pred_id]
        if label != "O":  # skip non-entity tokens
            entities.append({"token": token, "entity": label})

    return {"text": input.text, "entities": entities}
