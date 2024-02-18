import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, AutoTokenizer, AutoModelForCausalLM, DataCollatorWithPadding
import pandas as pd
from datasets import load_dataset

class MyModel:
    def __init__(self):
        self.user_context_map = {}
        self.tokenizer = None
        self.model = None
        self.trained_model = None

    def _reset_user_context(self, user_id):
        self.user_context_map[user_id] = torch.tensor([])

    def initialize_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.tokenizer.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        
    def build(self):
        # Initialize tokenizer and model if not already done
        if self.tokenizer is None or self.model is None:
            self.initialize_model()
            self.tokenizer.padding_side = "left"

        raw_datasets = load_dataset("Unified-Language-Model-Alignment/Anthropic_HH_Golden")
        tokenized_datasets = raw_datasets.map(self.tokenize_function, batched=True)
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        training_args = TrainingArguments("test-trainer")
        self.tokenizer.padding_side = "left"
        self.trained_model = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
        self.trained_model.train()

    def tokenize_function(self, examples):
        self.tokenizer.padding_side = "left"
        return self.tokenizer(examples["text"], padding="max_length", truncation=True)

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        reset_context = df["reset_context"].iloc[0]
        user_id = df["user_id"].iloc[0]
        message = df["message"].iloc[0]
        new_user_input_ids = self.tokenizer.encode(message + self.tokenizer.eos_token, return_tensors='pt')

        if reset_context:
            self._reset_user_context(user_id)

        bot_input_ids = torch.cat([self.user_context_map.get(user_id, torch.tensor([], dtype=torch.int64).reshape(1, -1)), new_user_input_ids], dim=-1)

        chat_history_ids = self.model.generate(bot_input_ids, max_length=1000, pad_token_id=self.tokenizer.eos_token_id)
        response = self.tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
        self.user_context_map[user_id] = chat_history_ids

        return pd.DataFrame(data={"answer": [response], "model_type": ["gpt2"]})

# Example usage:
my_model = MyModel()
my_model.initialize_model()  # Initialize model and tokenizer
# If you want to train the model, uncomment the next line and ensure you have defined tokenize_function correctly
# my_model.build()

# Making a prediction
df_input = pd.DataFrame(data={
    "user_id": ["Pavel"],
    "message": [" just random words"],
    "reset_context": [False]
})

response_df = my_model.predict(df_input)
print(response_df)