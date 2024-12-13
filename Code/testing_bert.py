import pandas as pd
from transformers import BertTokenizerFast, BertForQuestionAnswering
import torch

# Load tokenizer and model for mBERT
tokenizer = BertTokenizerFast.from_pretrained("output/fine_tuned_mbert_qa_20")
model = BertForQuestionAnswering.from_pretrained("output/fine_tuned_mbert_qa_20")

# Load the CSV file
input_df = pd.read_csv("input_data_evaluation_en.csv", sep=";")

# Function to predict answer
def predict_answer(text, question):
    inputs = tokenizer(
        question,
        text,
        return_tensors="pt",
        truncation="only_second",
        max_length=384,
        stride=128,
        return_overflowing_tokens=False,
        padding="max_length"
    )
    
    # Get model outputs
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Extract start and end logits
    start_logits = outputs.start_logits
    end_logits = outputs.end_logits
    
    # Find the start and end of the answer span
    answer_start = torch.argmax(start_logits, dim=1).item()
    answer_end = torch.argmax(end_logits, dim=1).item() + 1
    
    # Decode the answer
    answer = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end])
    )
    
    return answer

# Make predictions for each row
input_df["Answer"] = input_df.apply(lambda row: predict_answer(row["Text"], row["Question"]), axis=1)

# Save the result to a new CSV file
input_df.to_csv("results/results_en_bert_20.csv", sep=";", index=False)
