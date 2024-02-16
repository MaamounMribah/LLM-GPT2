from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
import torch
import pickle
import sys

def evaluate_model(model_path, test_data_path):
    model = GPT2LMHeadModel.from_pretrained(model_path)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    with open(test_data_path, 'rb') as f:
        test_dataset = pickle.load(f)

    training_args = TrainingArguments(
        output_dir='/tmp/',
        per_device_eval_batch_size=4,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=test_dataset,
    )

    eval_results = trainer.evaluate()

    evaluation_result_path = '/mnt/data/evaluation_results.txt'
    with open(evaluation_result_path, 'w') as f:
        for key, value in eval_results.items():
            f.write(f"{key}: {value}\n")

    print(f"Model evaluated. Results saved to {evaluation_result_path}")

if __name__ == '__main__':
    model_path = sys.argv[1]
    test_data_path = sys.argv[2]
    evaluate_model(model_path, test_data_path)
