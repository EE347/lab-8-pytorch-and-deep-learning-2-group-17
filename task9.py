import torch
import cv2
import numpy as np
from picamera2 import Picamera2
from torchvision.transforms import functional as F
from torchvision.models import mobilenet_v3_small
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Load the best model from Task 8
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = mobilenet_v3_small(num_classes=2)
model.load_state_dict(torch.load('/home/pi/ee347/lab-8-pytorch-and-deep-learning-2-group-17/best_model_final.pth'))
model.to(device)
model.eval()

# Initialize the Picamera2
picam2 = Picamera2()
picam2.configure(picam2.create_still_configuration(main={"size": (640, 480)}))
picam2.start()

# Collect predictions and actual labels for evaluation
all_preds = []
all_labels = []  # Add actual labels if available for specific test cases

try:
    print("Press Ctrl+C to stop the capture and evaluate.")

    while True:
        # Capture an image from the camera
        frame = picam2.capture_array()
       
        # Convert the image to a format suitable for the model
        image = cv2.resize(frame, (64, 64))  # Resize to model input size
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        image_tensor = F.to_tensor(image).unsqueeze(0).to(device)  # Normalize and add batch dimension

        # Perform inference
        with torch.no_grad():
            outputs = model(image_tensor)
            _, predicted = torch.max(outputs, 1)

        # Store predictions for confusion matrix if actual labels are available
        all_preds.append(predicted.item())

        # Display the prediction
        label = predicted.item()  # Assuming 0 and 1 are the class indices
        cv2.putText(frame, f'Predicted: {label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow('Camera - Press Q to Quit', frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    picam2.stop()
    cv2.destroyAllWindows()

# Calculate and display the confusion matrix if labels are available
if all_labels:  # Only if you have ground truth labels for evaluation
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(all_labels))
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix - Final Evaluation with Camera')
    plt.savefig('/home/pi/ee347/lab-8-pytorch-and-deep-learning-2-group-17/final_evaluation_confusion_matrix.png')
    plt.show()

# Print a message indicating completion
print("Inference completed and confusion matrix saved.")