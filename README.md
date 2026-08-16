# CNN_image_classification_final_project

# CNN-Based Fashion Image Classification Using PyTorch

## 1. Project Overview

This project implements a Convolutional Neural Network (CNN) for image classification using PyTorch.

The primary objective of this project is to train a CNN model on the Fashion-MNIST dataset and evaluate its performance on the standard test dataset. In addition, the trained model is tested on 10 real-world photographs captured using a smartphone.

The project demonstrates a complete deep learning workflow, including dataset acquisition, image preprocessing, model development, training, validation, evaluation, model saving, real-world prediction, and error analysis.

The project was developed using Python and PyTorch and executed in Google Colab.


## 2. Objectives

The main objectives of this project are:

- To implement a CNN using PyTorch.
- To train the CNN on the Fashion-MNIST dataset.
- To perform appropriate image preprocessing.
- To divide the training data into training and validation sets.
- To evaluate the model using the standard Fashion-MNIST test dataset.
- To visualize training and validation performance.
- To generate a confusion matrix.
- To test the trained model on real-world smartphone images.
- To analyze incorrectly classified test images.
- To save and organize the trained model and project files in a GitHub repository.

These objectives follow the requirements of the assignment, including standard-dataset training followed by real-world testing using custom photographs. :contentReference[oaicite:1]{index=1}


## 3. Dataset

### 3.1 Fashion-MNIST

The model is trained using the Fashion-MNIST dataset, obtained directly through `torchvision.datasets`.

Fashion-MNIST contains grayscale images belonging to 10 different clothing and fashion categories.

The classes are:

| Label | Class |
|------:|-------|
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

The dataset contains 60,000 training images and 10,000 test images. Each image has a resolution of 28 × 28 pixels and is represented as a grayscale image.


### 3.2 Custom Smartphone Dataset

For real-world evaluation, 10 photographs of fashion-related objects were captured using a smartphone.

The custom images are stored in the `dataset/` directory of this repository.

The custom dataset contains examples corresponding to the Fashion-MNIST classes selected for the real-world testing task.

The images were placed on a relatively plain background to reduce unnecessary visual noise before being photographed.

According to the assignment requirements, the custom images are processed to match the format of the training data, including grayscale conversion and resizing to 28 × 28 pixels. :contentReference[oaicite:2]{index=2}


### Custom Dataset Structure

```text
dataset/
├── sandal.jpg
├── sandal2.jpg
├── bag.jpg
├── bag2.jpg
├── sneaker.jpg
├── sneaker2.jpg
├── tshirt.jpg
├── tshirt2.jpg
├── trouser.jpg
└── trouser2.jpg



![Training and Validation Loss](images/5.png.png)

### 4. Data Preprocessing

The Fashion-MNIST images are processed using torchvision.transforms.

The preprocessing pipeline consists of:

Resizing the image to 28 × 28 pixels.
Converting the image to a PyTorch tensor.
Normalizing the image using the specified mean and standard deviation.

📷 INSERT IMAGE HERE — Custom Dataset
📷 INSERT IMAGE HERE — Custom Dataset



### 5. Training and Validation Split

The original Fashion-MNIST training dataset was divided into training and validation subsets.

The split used in this project is:

📷 INSERT IMAGE HERE — Custom Dataset


### 6. DataLoader

PyTorch DataLoader objects were created for the training, validation, and test datasets.

📷 INSERT IMAGE HERE — Custom Dataset


## 7. CNN Architecture

A custom Convolutional Neural Network was implemented using PyTorch.

The architecture consists of two convolutional layers followed by max-pooling operations and fully connected layers.
Input
28 × 28 × 1
      │
      ▼
Conv2D
1 → 32 channels
3 × 3 kernel
      │
      ▼
ReLU
      │
      ▼
MaxPool2D
      │
      ▼
Conv2D
32 → 64 channels
3 × 3 kernel
      │
      ▼
ReLU
      │
      ▼
MaxPool2D
      │
      ▼
Flatten
      │
      ▼
Fully Connected
3136 → 128
      │
      ▼
ReLU
      │
      ▼
Fully Connected
128 → 10
      │
      ▼
Output
10 Classes


📷 INSERT IMAGE HERE — Custom Dataset

8. Training Configuration

The following configuration was used for model training:
The training loop performs gradient clearing, forward propagation, loss calculation, backpropagation, and parameter updates using optimizer.zero_grad(), loss.backward(), and optimizer.step().


9. Experimental Results
9.1 Training and Validation Loss

The training and validation loss were recorded after each epoch to monitor the learning behavior of the model.

📊 INSERT GRAPH HERE — Loss vs Epoch 7
 
9.2 Training and Validation Accuracy

Training and validation accuracy were also recorded for every epoch.

📊 INSERT GRAPH HERE — Accuracy vs Epoch

10. Test Performance

After training, the model was evaluated on the standard Fashion-MNIST test dataset.

📊 INSERT GRAPH HERE — Loss vs Epoch 9


11. Confusion Matrix

A confusion matrix was generated using predictions from the standard Fashion-MNIST test dataset.

The confusion matrix provides a class-by-class comparison between the true labels and predicted labels.

It helps identify which categories are correctly classified and which categories are frequently confused by the model.

📊 INSERT IMAGE HERE — Confusion Matrix

📊 INSERT IMAGE HERE — Confusion Matrix


12. Custom Smartphone Image Prediction

The trained CNN was tested on 10 custom smartphone photographs.

The custom images were loaded from the GitHub repository using Python and processed using the same general preprocessing pipeline used for the training images.

For each custom image, the model produces:

Predicted class
Prediction confidence

13. Error Analysis

Three incorrectly classified images from the standard Fashion-MNIST test set were selected for visual error analysis.

he custom smartphone image experiment provides an opportunity to examine the difference between performance on the standardized Fashion-MNIST dataset and performance on real-world photographs.

Fashion-MNIST images have a controlled format, resolution, and grayscale representation. Smartphone photographs, on the other hand, may contain variations in:

Background
Lighting
Object orientation
Scale
Image quality
Visual complexity

Therefore, the performance on real-world images may differ from the performance obtained on the standard Fashion-MNIST test set.


15. Model Saving

After training, the model's learned parameters were saved using PyTorch's state_dict().

The trained model is stored as:

model/
└── 220150.pth
15. Model Saving

After training, the model's learned parameters were saved using PyTorch's state_dict().

The trained model is stored as:

model/
└── 220150.pth

16. Repository Structure

The final repository is organized as follows:

CNN-Image-Classification/
│
├── dataset/
│   ├── sandal.jpg
│   ├── sandal2.jpg
│   ├── bag.jpg
│   ├── bag2.jpg
│   ├── sneaker.jpg
│   ├── sneaker2.jpg
│   ├── tshirt.jpg
│   ├── tshirt2.jpg
│   ├── trouser.jpg
│   └── trouser2.jpg
│
├── model/
│   └── 220150.pth
│
├── images/
│   ├── custom_dataset.png
│   ├── loss_curve.png
│   ├── accuracy_curve.png
│   ├── confusion_matrix.png
│   ├── custom_predictions.png
│   └── error_analysis.png
│
├── 220150.ipynb
│
└── README.md

17. Technologies Used
Python
PyTorch
Torchvision
NumPy
Matplotlib
Scikit-learn
PIL
Google Colab
GitHub

18. Result
 পিচ ১২



19. Conclusion

This project demonstrates the implementation of a complete CNN-based image classification pipeline using PyTorch and the Fashion-MNIST dataset.

The project includes dataset acquisition, preprocessing, training and validation, CNN model development, performance evaluation, confusion matrix analysis, model saving, and real-world testing using custom smartphone photographs.

The results demonstrate the capability of a CNN to learn visual patterns from the Fashion-MNIST dataset and provide predictions for previously unseen images.

The project also highlights the importance of evaluating machine learning models using both standardized benchmark datasets and real-world examples.


