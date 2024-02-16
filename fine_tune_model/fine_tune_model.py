import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
import pickle
import sys

def fine_tune_model(preprocessed_data_path):
    preprocessed_data_path='preprocessed_data.pkl' 
    with open(preprocessed_data_path, 'rb') as f:
        tokenized_datasets = pickle.load(f)

    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    # Assuming tokenized_datasets is a dictionary with a 'train' key
    train_dataset = tokenized_datasets['train']
    
    training_args = TrainingArguments(
        output_dir='/mnt/data/gpt2_finetuned',
        num_train_epochs=3,
        per_device_train_batch_size=4,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='/logs',
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    trainer.train()

    model_path = '/mnt/data/gpt2_finetuned/model'
    model.save_pretrained(model_path)

    print(f"Model fine-tuned and saved to {model_path}")
    return (model_path,)


if __name__ == '__main__':
    preprocessed_data_path='./preprocessed_data.pkl' 
    fine_tune_model(preprocessed_data_path)
