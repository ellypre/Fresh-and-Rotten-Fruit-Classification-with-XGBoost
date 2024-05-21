
import numpy as np
import matplotlib.pyplot as plt
import glob
import cv2
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

# Set seed for reproducibility
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

# Load and preprocess data
def load_data(directory):
    images = []
    labels = []
    for folder in os.listdir(directory):
        label = folder.split('_')[-1]  # Extract label from folder name
        for file in glob.glob(os.path.join(directory, folder, "*.png")):
            img = cv2.imread(file)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # OpenCV reads images in BGR format, convert to RGB
            img = cv2.resize(img, (128, 128))
            images.append(img)
            labels.append(label)
    return np.array(images), np.array(labels)

train_images, train_labels = load_data("C:/HK/yapay zeka/dataset2/train")
test_images, test_labels = load_data("C:/HK/yapay zeka/dataset2/test")

# Encode labels
label_encoder = LabelEncoder()
train_labels_encoded = label_encoder.fit_transform(train_labels)
test_labels_encoded = label_encoder.transform(test_labels)

# Normalize pixel values and convert to PyTorch tensors
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
])

train_dataset = [(transform(image), label) for image, label in zip(train_images, train_labels_encoded)]
test_dataset = [(transform(image), label) for image, label in zip(test_images, test_labels_encoded)]

# Train/test split
train_dataset, val_dataset = train_test_split(train_dataset, test_size=0.2, random_state=seed)

# Define model architecture
class CustomVGG(nn.Module):
    def __init__(self, num_classes):
        super(CustomVGG, self).__init__()
        vgg = models.vgg16(pretrained=True)
        self.features = vgg.features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# Initialize model, loss function, and optimizer
model = CustomVGG(num_classes=len(label_encoder.classes_))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Convert datasets to DataLoader
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    epoch_loss = running_loss / len(train_dataset)
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}")

# Validation loop
model.eval()
val_correct = 0
with torch.no_grad():
    for images, labels in val_loader:
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        val_correct += (preds == labels).sum().item()
val_accuracy = val_correct / len(val_dataset)
print(f"Validation Accuracy: {val_accuracy:.4f}")

# Test loop
model.eval()
test_features = []
test_labels = []
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        test_features.extend(outputs.numpy())
        test_labels.extend(labels.numpy())

# Train XGBoost model
xgb_model = XGBClassifier(tree_method='gpu_hist', random_state=seed)  # Use GPU
xgb_model.fit(test_features, test_labels)

# Predictions
train_preds = xgb_model.predict(test_features)
train_accuracy = accuracy_score(test_labels, train_preds)
print("Test Accuracy (XGBoost):", train_accuracy)

# Confusion Matrix
cm = confusion_matrix(test_labels, train_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', cbar=False, xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix (XGBoost)')
plt.show()
