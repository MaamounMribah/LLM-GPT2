
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import load_dataset

# Model Fine-Tuning Component
def fine_tune_model():
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.resize_token_embeddings(len(tokenizer))
    # Load dataset and split into train and eval
    #raw_datasets = load_dataset('Unified-Language-Model-Alignment/Anthropic_HH_Golden', split={'train': 'train[:1%]', 'eval': 'test[:1%]'})
    
    
    def preprocess_function(examples):
        result = tokenizer(examples['text'], padding='max_length', truncation=True, max_length=128)
        result["labels"] = result["input_ids"][:]
        return result

    raw_datasets = load_dataset('ag_news', split={'train_subset': 'train[:1%]', 'test_subset': 'test[:1%]'})
    tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)
    
    
    train_dataset = tokenized_datasets['train_subset']
    eval_dataset = tokenized_datasets['test_subset']

    training_args = TrainingArguments(
        output_dir='./gpt2_finetuned',
        num_train_epochs=3,
        per_device_train_batch_size=8,  # Reduced batch size
        gradient_accumulation_steps=4,  # Implement gradient accumulation
        
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        load_best_model_at_end=True,
        evaluation_strategy="epoch",
        save_strategy="epoch"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,  # Provide the eval_dataset here
    )  

    trainer.train()

    model_path = './gpt2_finetuned'
    model.save_pretrained(model_path)

    print(f"Model fine-tuned and saved to {model_path}")
    

if __name__ == '__main__':
    
    fine_tune_model()