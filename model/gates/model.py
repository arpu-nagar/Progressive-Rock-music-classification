import torch
prog_tensors = torch.load('../../content/progressive_rock_songs_tensor.pt')
non_prog_tensors = torch.load('../../content/non_progressive_rock_songs_tensor.pt')

print(prog_tensors.shape)
print(non_prog_tensors.shape)

# prompt: non_prog_tensors reshaped to [8052, 160, 216])

non_prog_tensors = non_prog_tensors.reshape(8052, 160, 216)
prog_tensors = prog_tensors.reshape(7455, 160, 216)
prog_tensors = prog_tensors.float()
non_prog_tensors = non_prog_tensors.float()

from sklearn.model_selection import train_test_split

# Concatenate the tensors and create labels
data = torch.cat((prog_tensors, non_prog_tensors), dim=0)
labels = torch.cat((torch.ones(prog_tensors.shape[0]), torch.zeros(non_prog_tensors.shape[0])), dim=0)

# the answer to life, universe and anything
# train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.5, random_state=42)
# train_data, val_data, train_labels, val_labels = train_test_split(train_data, train_labels, test_size=0.5, random_state=42)

batch_size = 32  # Define your desired batch size
data.shape
# (instances, timesteps, features) [15507, 216, 160]

import torch
from torch.utils.data import TensorDataset, DataLoader

# Create TensorDataset from your data and labels
train_dataset = TensorDataset(data, labels)
# val_dataset = TensorDataset(val_data, val_labels)
# test_dataset = TensorDataset(test_data, test_labels)

# Create DataLoader for each dataset with the specified batch size
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

import torch
import torch.nn as nn

class MyConvNet(nn.Module):
    def __init__(self):
        super(MyConvNet, self).__init__()

        # Define convolutional layers
        self.conv1 = nn.Conv1d(in_channels=160, out_channels=240, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=240, out_channels=360, kernel_size=5, stride=1, padding=1)
        self.conv3 = nn.Conv1d(in_channels=360, out_channels=480, kernel_size=10, stride=1, padding=1)
        self.conv4 = nn.Conv1d(in_channels=480, out_channels=256, kernel_size=5, stride=1, padding=1)
        self.conv5 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)

        # Define fully connected layers
        self.fc1 = nn.Linear(205*64, 200)  # Assuming input size after convolutions is 6656 (13*512)
        self.fc2 = nn.Linear(200, 10)
        self.fc3 = nn.Linear(10, 2)

        # Define activation function
        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(p=0.25)

    def forward(self, x):
        # Convolutional layers with ReLU activation
        x = self.relu(self.conv1(x))
        x = self.dropout(x)
        x = self.relu(self.conv2(x))
        x = self.dropout(x)
        x = self.relu(self.conv3(x))
        x = self.dropout(x)
        x = self.relu(self.conv4(x))
        x = self.dropout(x)
        x = self.relu(self.conv5(x))
        x = self.dropout(x)
        x = self.relu(self.conv6(x))
        x = self.dropout(x)

        # Flatten the output of the convolutional layers
        x = x.view(x.size(0), -1)

        # Fully connected layers with ReLU activation
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        # x = self.dropout(x)
        x = self.fc3(x)

        return x

# Instantiate the model
model = MyConvNet()
print(model)

# Define loss function
criterion = nn.CrossEntropyLoss()

# Define optimizer (e.g., Adam optimizer with learning rate 0.001)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

print(model)
device = "cuda:0"
model.to(device)
model.train()
loss_hist = []
correct_predictions = 0
total_predictions = 0
true_positives = 0
false_positives = 0
true_negatives = 0
false_negatives = 0
epochs = 15

for e in range(epochs):
    i = 1
    loss_per_epoch = 0
    for inputs, labels in train_loader:
        # Perform forward pass, compute loss, and update the model
        inputs = inputs.unsqueeze(1)
        labels = labels.long()
        bs, c, time, feats = inputs.shape
        inputs = inputs.reshape(bs, time, feats)
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        outputs = model(inputs)

        # Calculate loss
        loss = criterion(outputs, labels)
        
        loss_per_epoch += loss.item()
        # Calculate accuracy
        # print('a',outputs.data)
        # print('b',torch.max(outputs.data, 1))
        _, predicted = torch.max(outputs.data, 1)
        total_predictions += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()

        # Calculate precision
        true_positives += ((predicted == 1) & (labels == 1)).sum().item()
        false_positives += ((predicted == 1) & (labels == 0)).sum().item()
        true_negatives += ((predicted == 0) & (labels == 0)).sum().item()
        false_negatives += ((predicted == 0) & (labels == 1)).sum().item()

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print metrics
        print("Epoch: {}, Batch: {}, Loss: {:.4f}, Accuracy: {:.2f}%, Precision: {:.2f}%".format(
            e, i, loss.item(), (correct_predictions / total_predictions) * 100,
            (true_positives / (true_positives + false_positives + 1e-12)) * 100  # Add small epsilon to avoid division by zero
        ))

        i += 1
    epoch_loss = loss_per_epoch/i
    loss_hist.append(epoch_loss)
    writer.add_scalar("Loss/train_model", epoch_loss, e)
    writer.flush()

torch.save(model.state_dict(), "model.pt")