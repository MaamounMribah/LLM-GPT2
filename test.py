import pickle
from datasets import load_dataset
from torch.utils.data import DataLoader

# Replace 'preprocessed_data_path' with the actual path to your pickle file
preprocessed_data_path = 'preprocessed_data.pkl'

with open(preprocessed_data_path, 'rb') as f:
    data = pickle.load(f)

# Print the type and optionally some content to understand its structure
print(type(data))
#print(data.keys()) if isinstance(data, dict) else print(data)
print(data)
dataset = load_dataset('ag_news', 'default', split='train[:1%]')

# Assuming 'dataset' is your PyTorch Dataset object
data_loader = DataLoader(dataset, batch_size=16, shuffle=True)

for batch in data_loader:
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']
    # If you're in a supervised learning context
    labels = batch.get('labels', None)  # Using .get() to avoid KeyError if 'labels' key doesn't exist
    print(input_ids.shape)

