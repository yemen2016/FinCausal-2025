import pandas as pd
from datasets import Dataset
from transformers import BertTokenizerFast, BertForQuestionAnswering, Trainer, TrainingArguments

# Load CSV files
train_df = pd.read_csv("training_all.csv", sep=",", on_bad_lines="skip")
dev_df = pd.read_csv("devlopment_all.csv", sep=",", on_bad_lines="skip")

# Function to find answer's start position in the context
def find_answer_start(context, answer):
    return context.find(answer)

# Convert DataFrames to list of dictionaries for JSON-like structure
def convert_to_qa_format(df):
    qa_data = []
    for _, row in df.iterrows():
        context = row['Text']
        question = row['Question']
        answer = row['Answer']
        start_position = find_answer_start(context, answer)
        
        if start_position != -1:
            qa_data.append({
                "id": row['ID'],
                "context": context,
                "question": question,
                "answers": {
                    "text": [answer],
                    "answer_start": [start_position]
                }
            })
    return qa_data

# Convert dataframes to QA format
train_data = convert_to_qa_format(train_df)
dev_data = convert_to_qa_format(dev_df)

# Load as Hugging Face Dataset objects
train_dataset = Dataset.from_list(train_data)
dev_dataset = Dataset.from_list(dev_data)

# Use the tokenizer for mBERT
tokenizer = BertTokenizerFast.from_pretrained("bert-base-multilingual-cased")

# Preprocessing function for tokenization
def preprocess_function(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        truncation="only_second",
        max_length=384,
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # Mapping examples to features
    sample_mapping = inputs.pop("overflow_to_sample_mapping")
    offset_mapping = inputs.pop("offset_mapping")

    start_positions = []
    end_positions = []

    for i, offsets in enumerate(offset_mapping):
        input_ids = inputs["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        # Example index
        sequence_ids = inputs.sequence_ids(i)
        example_index = sample_mapping[i]
        answers = examples["answers"][example_index]
        answer_starts = answers["answer_start"]
        answer_texts = answers["text"]

        if len(answer_starts) == 0:
            start_positions.append(cls_index)
            end_positions.append(cls_index)
        else:
            start_char = answer_starts[0]
            end_char = start_char + len(answer_texts[0])

            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1

            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != 1:
                token_end_index -= 1

            if offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char:
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                start_positions.append(token_start_index - 1)

                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                end_positions.append(token_end_index + 1)
            else:
                start_positions.append(cls_index)
                end_positions.append(cls_index)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

# Apply preprocessing function
tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True, remove_columns=train_dataset.column_names)
tokenized_dev_dataset = dev_dataset.map(preprocess_function, batched=True, remove_columns=dev_dataset.column_names)

# Load mBERT model for question answering
model = BertForQuestionAnswering.from_pretrained("bert-base-multilingual-cased")

# Training arguments
training_args = TrainingArguments(
    output_dir="results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=20,
    weight_decay=0.01,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_dev_dataset,
    tokenizer=tokenizer,
)

# Start training
trainer.train()

# Evaluate
results = trainer.evaluate()
print(results)

# Save model and tokenizer
model.save_pretrained("output/fine_tuned_mbert_qa_20")
tokenizer.save_pretrained("output/fine_tuned_mbert_qa_20")
