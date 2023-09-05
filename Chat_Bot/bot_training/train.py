if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()

import json
import numpy as np
from nltk_utils import tokenize, stem, bag_of_words
from model import NeuralNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

with open('Resources\indents.json', 'r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []
for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

ignore_words = ['?', '!', '#', ';', ':', 'newlinechar', '.', ',', '@', '^', '*', '+', '=', '-', '~', '`', '/', 'fuck', 'fucking', 'shit', 'shitting', 'bitch', 'ass']
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

X = []
y = []
for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X.append(bag)
    label = tags.index(tag)
    y.append(label)

X = np.array(X)
y = np.array(y)

# Perform the train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.n_samples
    
# Hyperparameters
batch_size = 8
hidden_size = 8
output_size = len(tags)
input_size = len(X_train[0])
learning_rate = 0.001
num_epochs = 2500

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size, hidden_size, output_size)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training the model
model.train()
num_epochs = 2500
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(device=device, dtype=torch.int64)

        # Forward
        outputs = model(words)
        loss = criterion(outputs, labels)

        # Backward and optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch+1) % 100 == 0:
        print(f'epoch {epoch+1}/{num_epochs}, loss={loss.item():.4f}')

# Evaluating the model
model.eval()
with torch.no_grad():
    test_inputs = torch.tensor(X_test).to(device)
    predicted_labels = model(test_inputs)
    _, predicted_labels = torch.max(predicted_labels, dim=1)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, predicted_labels.cpu().numpy())
precision = precision_score(y_test, predicted_labels.cpu().numpy(), average='weighted')
recall = recall_score(y_test, predicted_labels.cpu().numpy(), average='weighted')
f1 = f1_score(y_test, predicted_labels.cpu().numpy(), average='weighted')

print(f'Test Accuracy: {accuracy:.4f}')
print(f'Test Precision: {precision:.4f}')
print(f'Test Recall: {recall:.4f}')
print(f'Test F1 Score: {f1:.4f}')
print(f'Final Loss: {loss.item():.4f}')

# Saving the model and relevant data
data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "output_size": output_size,
    "hidden_size": hidden_size,
    "all_words": all_words,
    "tags": tags,
    "test_data": {
        "X_test": X_test,
        "y_test": y_test,
    },
    "evaluation_metrics": {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
}

file = "data.pth"
torch.save(data, file)

print(f'Training complete. File saved to {file}')