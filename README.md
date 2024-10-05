# Facial Recognition using VGG16

## Objective
The goal of this project is to develop a facial recognition model using the **VGG16** architecture, pre-trained on ImageNet, to classify images from the ORL dataset.

## Dataset
- **ORL Database**: 
  - 240 training images, 160 test images.
  - Each image is resized to **224x224x3** (RGB) to fit the input requirements for the VGG16 model.
  
## Data Preprocessing
- **Image Reshaping and Normalization**:
  - Grayscale images are resized and converted to RGB.
  - Images are normalized by dividing pixel values by 255.
  
- **One-Hot Encoding**:
  - Labels for the 20 subjects are converted into one-hot encoded vectors.
  
- **Data Augmentation**:
  - Applied rotation, width and height shifts, and horizontal flips to enhance model generalization.

## Model Architecture
- **Base Model**: 
  - **VGG16** pre-trained on ImageNet, with the last few layers fine-tuned.
  
- **Additional Layers**:
  - Fully connected layer with 512 units and **ReLU** activation.
  - **Dropout** and **Batch Normalization** layers to prevent overfitting.
  - Final **softmax** layer for classification into 20 classes (20 subjects).

## Training and Results
- **Optimization**: 
  - Used **Adam optimizer** with a learning rate of `1e-4`.
  
- **Epochs**: 
  - Trained the model for **35 epochs** with data augmentation.

- **Training Performance**:
  - Initial accuracy: 6.36% (Epoch 1).
  - Steady improvement over epochs.
  - Final validation accuracy: **100%** (Epoch 21 onwards).
  - Test accuracy: **100%** after training.

## Evaluation
- **Test Accuracy**: The model achieved 100% accuracy on the test set, perfectly classifying all images.
- **Loss Curves**: Both training and validation loss consistently decreased, reflecting model convergence without significant overfitting.

## Inferences
- **Highly Accurate Model**: The final test accuracy of 100% demonstrates the model's ability to perfectly recognize and classify faces from the ORL dataset.
- **Generalization**: Data augmentation techniques and fine-tuning contributed to strong generalization on unseen data.
- **Pre-trained Model Efficiency**: Using a pre-trained VGG16 model significantly boosted performance, allowing faster convergence and higher accuracy.

## Conclusion
This model is highly effective for the specific task of recognizing faces from a fixed set of individuals and can serve as a robust baseline for similar facial recognition tasks.
