# After build your Biomedical NER system

# Terminal: pip install fastapi uvicorn transformers

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

# Initialize the FastAPI app
app = FastAPI()

# Load the pre-trained BioBERT model and tokenizer
model_name = "dmis-lab/biobert-v1.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

# Create a pipeline for Named Entity Recognition (NER)
nlp = pipeline("ner", model=model, tokenizer=tokenizer)

# Define the input data model using Pydantic
class TextInput(BaseModel):
    text: str

# Define the route for NER
@app.post("/ner/")
async def extract_entities(text_input: TextInput):
    text = text_input.text
    # Get NER results
    ner_results = nlp(text)
    return {"entities": ner_results}

