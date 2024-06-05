import pandas as pd
import torch
# Normalize/Standardize data if necessary
from sklearn.preprocessing import StandardScaler
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

file_path = r'C:\Users\akshat.priyadarshan\OneDrive - Accenture\Python\GenAI\temp_data.csv'

# Try different encodings to find the one that works
encodings = ['utf-8', 'latin1', 'ISO-8859-1', 'cp1252']
for encoding in encodings:
    try:
        data = pd.read_csv(file_path, encoding=encoding)
        print(f"Successfully read the CSV file with {encoding} encoding")
        break
    except UnicodeDecodeError:
        print(f"Failed to read the CSV file with {encoding} encoding")
else:
    raise ValueError("Unable to read the CSV file with the provided encodings")

# Convert the data to a text format for training
texts = []
for idx, row in data.iterrows():
    record_str = ', '.join([f"{k}: {v}" for k, v in row.items()])
    texts.append(f"Record {idx}: {record_str}\n")

# Save to a text file
with open('fine_tuning_data.txt', 'w') as f:
    for text in texts:
        f.write(text + "\n")
""" 
scaler = StandardScaler()
numeric_columns = data.select_dtypes(include=['float64', 'int']).columns
data[numeric_columns] = scaler.fit_transform(data[numeric_columns])

# Save preprocessed data for later use
data.to_csv('preprocessed_data.csv', index=False) """

# Load pre-trained GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Prepare the dataset
def load_dataset(file_path, tokenizer, block_size=128):
    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=block_size
    )
    return dataset

# Load the dataset
train_dataset = load_dataset('fine_tuning_data.txt', tokenizer)

# Initialize data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# Set training arguments
training_args = TrainingArguments(
    output_dir='./results',
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    save_steps=10_000,
    save_total_limit=2,
)

# Initialize the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
)

# Train the model
trainer.train()

# Save the model
model.save_pretrained('./fine_tuned_model')
tokenizer.save_pretrained('./fine_tuned_model')

# Load the fine-tuned model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('./fine_tuned_model')
model = GPT2LMHeadModel.from_pretrained('./fine_tuned_model')

# Function to generate response with the fine-tuned model
def generate_response(prompt, record=None):
    if record:
        record_str = ', '.join([f"{k}: {v}" for k, v in record.items()])
        prompt = f"The record contains the following details: {record_str}. Based on this information, provide relevant insights."
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    attention_mask = torch.ones(inputs.shape, dtype=torch.long)
    
    outputs = model.generate(
        inputs, 
        max_length=150, 
        num_return_sequences=1,
        attention_mask=attention_mask,
        pad_token_id=tokenizer.eos_token_id
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def interactive_chat(data):
    print("Welcome to the CSV Chatbot! Ask me about the data.")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            break
        if user_input.lower().startswith("tell me about record"):
            try:
                record_id = int(user_input.split()[-1])
                record = data.iloc[record_id].to_dict()
                response = generate_response("", record=record)
                print(f"Bot: {response}")
            except Exception as e:
                print(f"Error: {e}")
        else:
            response = generate_response(user_input)
            print(f"Bot: {response}")

# Load preprocessed data
data = pd.read_csv('preprocessed_data.csv')
interactive_chat(data)
