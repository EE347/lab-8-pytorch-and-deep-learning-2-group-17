import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import DataLoader
from utils.dataset import TeamMateDataset

# Specify device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the best saved model from Task 8
model = mobilenet_v3_small(num_classes=2)
model.load_state_dict(torch.load('/home/pi/ee347/lab-8-pytorch-and-deep-learning-2-group-17/best_model_final.pth'))
model.to(device)
model.eval()

# Load the dataset for evaluation (test set or a new dataset)
testset = TeamMateDataset(n_images=10, train=False)  # Adjust the path and dataset parameters if needed
testloader = DataLoader(testset, batch_size=1, shuffle=False)

# Lists to hold all predictions and actual labels for evaluation
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in testloader:
        images = images.reshape(-1, 3, 64, 64).to(device)
        labels = labels.to(device)

        # Forward pass to get model outputs
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        # Collect predictions and labels for evaluation
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Calculate accuracy
accuracy = accuracy_score(all_labels, all_preds)
print(f'Accuracy on the test dataset: {accuracy:.4f}')

# Generate and display the confusion matrix
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(all_labels))
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix - Final Evaluation')
plt.savefig('/home/pi/ee347/lab-8-pytorch-and-deep-learning-2-group-17/final_evaluation_confusion_matrix.png')
plt.show()