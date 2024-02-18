

from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import load_dataset

def evaluate_model():
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    model.resize_token_embeddings(len(tokenizer))
    
    # with open(test_data_path, 'rb') as f:
        # test_dataset = pickle.load(f)

    dataset = load_dataset('ag_news', split='test[:1%]')

    def preprocess_function(examples):
        result = tokenizer(examples['text'], padding='max_length', truncation=True, max_length=128)
        # For language modeling, the labels are the input_ids shifted by one
        result["labels"] = result["input_ids"][:]
        return result
        #return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=128, return_tensors="pt")

    tokenized_datasets = dataset.map(preprocess_function, batched=True)

    training_args = TrainingArguments(
        output_dir='/tmp/',
        per_device_eval_batch_size=4,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=tokenized_datasets,
    )

    eval_results = trainer.evaluate()

    evaluation_result_path = 'evaluation_result.txt'
    with open(evaluation_result_path, 'w') as f:
        for key, value in eval_results.items():
            f.write(f"{key}: {value}\n")

    print(f"Model evaluated. Results saved to {evaluation_result_path}")
    print("Evaluation Results:")
    for key, value in eval_results.items():
        print(f"{key}: {value}")

if __name__ == '__main__':
    evaluate_model()