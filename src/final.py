import torch
from transformers import BertTokenizer
from main import TextCNN
import argparse

# Define the argument parser
parser = argparse.ArgumentParser(description='Predict Positive or negative')
parser.add_argument('--text', type=str,
                    help='Text to decide positive or negative')

batch_size = 64
max_seq_len = 256
embedding_dim = 50
num_filters = 128
filter_sizes = [3, 4, 5]
output_dim = 1
dropout_prob = 0.5
learning_rate = 1e-3
num_epochs = 5

# Load the trained model
model_state_dict = torch.load('./models/model.pth')
model = TextCNN(30522, embedding_dim, num_filters,
                filter_sizes, dropout_prob)
model.load_state_dict(model_state_dict)

# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

args = parser.parse_args()
# Tokenize your own text

tokens = tokenizer.encode_plus(args.text, add_special_tokens=True, padding='max_length',
                               max_length=256, truncation=True, return_tensors='pt')

# Pass the tokens through the model
with torch.no_grad():
    outputs = model(input_ids=tokens['input_ids'],
                    attention_mask=tokens['attention_mask'])
# Get the predicted label
predicted_label = 'positive' if outputs[0][0] >= 0.9 else 'negative'
print(f'{predicted_label}')
