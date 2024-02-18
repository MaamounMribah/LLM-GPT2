from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset

def generate_text(model, tokenizer, test_dataset, num_samples=10, max_length=128):
    model.eval()  # Set the model to evaluation mode
    for i, example in enumerate(test_dataset):
        inputs = tokenizer.encode(example['text'], return_tensors='pt', max_length=max_length, truncation=True)
        # Generate text using the model
        output_sequences = model.generate(
            input_ids=inputs,
            max_length=max_length,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            top_p=0.95,
            top_k=50,
        )
        # Decode the generated text
        generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
        print(f"Original Text: {example['text']}")
        print(f"Generated Text: {generated_text}")
        if i >= num_samples - 1:
            break

def evaluate_model_without_finetuning():
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    dataset = load_dataset('ag_news', split='test[:1%]')
    
    generate_text(model, tokenizer, dataset)

if __name__ == '__main__':
    evaluate_model_without_finetuning()