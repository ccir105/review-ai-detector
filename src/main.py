import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
from transformers import BertTokenizer

# Define hyperparameters
batch_size = 64
max_seq_len = 256
embedding_dim = 50
num_filters = 128
filter_sizes = [3, 4, 5]
output_dim = 1
dropout_prob = 0.5
learning_rate = 1e-3
num_epochs = 5


class TextDataset(Dataset):
    def __init__(self, data, max_seq_len=256):
        self.texts = data['text']
        self.labels = data['label']
        self.max_seq_len = max_seq_len
        self.tokenizer = None
        self.vocab = None

        if self.tokenizer is None:
            self.tokenizer = torch.hub.load(
                'huggingface/pytorch-transformers', 'tokenizer', 'bert-base-uncased')
            self.vocab = self.tokenizer.get_vocab()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts.iloc[idx]
        label = self.labels.iloc[idx]

        # Tokenize the text and truncate or pad to max_seq_len
        tokens = self.tokenizer.encode(text, add_special_tokens=True)[
            :self.max_seq_len]
        if len(tokens) < self.max_seq_len:
            tokens += [self.vocab['[PAD]']] * (self.max_seq_len - len(tokens))
        tokens = torch.tensor(tokens)

        # Convert label to numerical value
        label = torch.tensor(1) if label == 'pos' else torch.tensor(0)

        return tokens, label


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_filters, filter_sizes, dropout_rate):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, kernel_size=f)
            for f in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(num_filters * len(filter_sizes), 1)

    def forward(self, input_ids, attention_mask=[]):
        # Embed the input tokens
        x = self.embedding(input_ids)  # [batch_size, seq_len, embed_dim]
        x = x.permute(0, 2, 1)  # [batch_size, embed_dim, seq_len]

        # Apply convolutional filters and pooling layers
        pooled_outputs = []
        for conv in self.convs:
            # [batch_size, num_filters, seq_len - filter_size + 1]
            conv_output = conv(x)
            pooled_output = nn.functional.max_pool1d(
                conv_output, kernel_size=conv_output.size(2)).squeeze(2)  # [batch_size, num_filters]
            pooled_outputs.append(pooled_output)
        # [batch_size, num_filters * len(filter_sizes)]
        x = torch.cat(pooled_outputs, dim=1)

        # Apply dropout and linear layer
        x = self.dropout(x)
        x = self.fc(x)  # [batch_size, 1]

        # Apply sigmoid activation function
        x = torch.sigmoid(x)
        return x


def train(model, optimizer, criterion, train_loader, val_loader, num_epochs):
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        # Train the model
        model.train()
        train_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets.float().unsqueeze(1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            print(
                f'Training batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')

        # Validate the model
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(val_loader):
                outputs = model(inputs)
                loss = criterion(outputs, targets.float().unsqueeze(1))
                val_loss += loss.item()
                val_preds.extend(outputs.round().squeeze().tolist())
                val_targets.extend(targets.tolist())
                print(
                    f'Evaluating batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')

        # Compute evaluation metrics
        val_loss /= len(val_loader)
        val_acc = accuracy_score(val_targets, val_preds)
        val_precision = precision_score(
            val_targets, val_preds, average='weighted')
        val_recall = recall_score(val_targets, val_preds, average='weighted')
        val_f1 = f1_score(val_targets, val_preds, average='weighted')

        # Print training and validation losses and metrics
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}, Val F1: {val_f1:.4f}')

        # Save the best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(os.getcwd())
            torch.save(model.state_dict(), 'models/model.pth')

    print("Training Completed")


def test(model, criterion, test_loader):
    model.eval()
    test_loss = 0.0
    test_preds = []
    test_targets = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            outputs = model(inputs)
            loss = criterion(outputs, targets.float().unsqueeze(1))
            test_loss += loss.item()
            test_preds.extend(outputs.round().squeeze().tolist())
            test_targets.extend(targets.tolist())

    # Compute evaluation metrics
    test_loss /= len(test_loader)
    test_acc = accuracy_score(test_targets, test_preds)
    test_precision = precision_score(test_targets, test_preds)
    test_recall = recall_score(test_targets, test_preds)
    test_f1 = f1_score(test_targets, test_preds)

    # Print testing loss and metrics
    print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, Test Precision: {test_precision:.4f}, Test Recall: {test_recall:.4f}, Test F1: {test_f1:.4f}')


def create_data_loader(data, batch_size=32, max_seq_len=512):
    dataset = TextDataset(data, max_seq_len=max_seq_len)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def main():
    # Load the preprocessed data
    train_data = pd.read_csv('csv/train_data.csv')
    test_data = pd.read_csv('csv/test_data.csv')

    dataset = TextDataset(pd.concat([train_data, test_data]))

    vocab_size = len(dataset.vocab)

    # Split the data into training and validation sets
    train_data, val_data = train_test_split(
        train_data, test_size=0.2, random_state=42)

    # Create data loaders for the training, validation, and testing data
    train_loader = create_data_loader(train_data)
    val_loader = create_data_loader(val_data)
    test_loader = create_data_loader(test_data)

    # Initialize the CNN model, optimizer, and loss function
    model = TextCNN(vocab_size, embedding_dim, num_filters,
                    filter_sizes, dropout_prob)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    # Train the model
    train(model, optimizer, criterion, train_loader, val_loader, num_epochs)

    # Evaluate the model on the testing data
    test(model, criterion, test_loader)


if __name__ == '__main__':
    main()
