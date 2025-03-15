# An example of your entity csv file:
# text,gene_entity,disease_entity,chemical_entity
#"BRCA1 mutations are associated with breast cancer and linked to chemotherapy resistance.","BRCA1","breast cancer","chemotherapy resistance"
#"TP53 gene is often mutated in various cancers.","TP53","cancer",""

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from seqeval.metrics import classification_report

# Step 1: Load the pre-trained PubMedBERT model and tokenizer
model_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-ner"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

# Step 2: Load the CSV file with entity annotations
csv_file = "entities.csv"  # Path to your CSV file
df = pd.read_csv(csv_file)

# Step 3: Function to run NER inference and compare with CSV entity labels
def run_ner_inference(text, true_labels):
    # Tokenize the input text
    tokens = tokenizer(text.split(), is_split_into_words=True, return_tensors="pt", truncation=True, padding=True)
    
    # Run inference
    with torch.no_grad():
        outputs = model(**tokens)

    # Get predicted labels (indices of tokens in the output)
    predictions = torch.argmax(outputs.logits, dim=2)

    # Decode the predictions back into words
    predicted_labels = predictions[0].tolist()

    # Map the labels to their names (using the tokenizer's label mapping)
    label_map = {i: label for i, label in enumerate(model.config.id2label.values())}
    labels = [label_map[label_id] for label_id in predicted_labels]

    # Combine tokens with their corresponding labels
    results = list(zip(tokens["input_ids"][0], labels))

    # Create a list of entities in the text
    predicted_entities = []
    for token, label in results:
        word = tokenizer.decode([token])
        if label != "O":  # Only collect non-"O" (non-entity) labels
            predicted_entities.append((word, label))

    # Compare the predicted entities with the true labels (from the CSV)
    return predicted_entities, true_labels

# Step 4: Prepare to collect results for saving to CSV
output_data = []

# Step 5: Loop through each row in the CSV and run inference
for _, row in df.iterrows():
    text = row['text']
    true_labels = []
    
    # Add entities to true labels based on columns for gene, disease, and chemical entities
    if pd.notna(row['gene_entity']):
        true_labels.append(('GENE', row['gene_entity']))
    if pd.notna(row['disease_entity']):
        true_labels.append(('DISEASE', row['disease_entity']))
    if pd.notna(row['chemical_entity']):
        true_labels.append(('CHEMICAL', row['chemical_entity']))

    # Run NER on the text and compare
    predicted_entities, true_labels = run_ner_inference(text, true_labels)

    # Prepare the row to be saved (Text, True Labels, Predicted Entities)
    true_entities_str = "; ".join([f"{label}: {entity}" for label, entity in true_labels])
    predicted_entities_str = "; ".join([f"{label}: {entity}" for entity, label in predicted_entities])

    # Append results to the output data list
    output_data.append({
        "Text": text,
        "True_Entities": true_entities_str,
        "Predicted_Entities": predicted_entities_str
    })

# Step 6: Save the results to a new CSV file
output_df = pd.DataFrame(output_data)
output_df.to_csv("ner_results.csv", index=False)

print("NER results saved to 'ner_results.csv'")


