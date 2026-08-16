# CNN_image_classification_final_project


 ## 1️⃣ Introduction

This project presents an end-to-end implementation of a Convolutional Neural Network (CNN) for Fashion-MNIST image classification using PyTorch.

The model is trained on the standard Fashion-MNIST dataset and evaluated using a separate test set. In addition, the trained model is tested on 10 real-world smartphone images to study how well the CNN generalizes beyond the standard dataset.

The project covers the complete deep learning workflow, including:

- Dataset loading
- Image preprocessing
- Training and validation split
- DataLoader creation
- CNN model development
- Model training
- Validation
- Test evaluation
- Loss and accuracy visualization
- Confusion matrix analysis
- Custom smartphone image prediction
- Error analysis


---

## 2️⃣ Dataset Used

### 2.1 Fashion-MNIST Dataset

The main dataset used in this project is the Fashion-MNIST dataset provided through `torchvision.datasets`.

Fashion-MNIST contains grayscale images of fashion items belonging to 10 different classes.

### Classes

| Label | Class |
|---|---|
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |


The dataset contains:

- 60,000 training images
- 10,000 test images
- 10 classes
- Image size: 28 × 28 pixels
- Image type: Grayscale


### 2.2 Custom Smartphone Dataset

A separate custom dataset containing 10 smartphone images was created for real-world evaluation.

These images are stored inside the `dataset/` directory of this repository.

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

## 3️⃣ Data Preprocessing

The same preprocessing pipeline is used to prepare the Fashion-MNIST images.

3.1 Training Dataset Transform

The following transformations are applied:

Resize to 28 × 28
Convert image to tensor
Normalize pixel values

<img width="429" height="142" alt="image" src="https://github.com/user-attachments/assets/63dcf2a0-538e-441d-959c-f88002f2c614" />

3.2 Custom Smartphone Image Transform

Since the smartphone images are real-world RGB images, they are converted to grayscale before being passed to the CNN.

The custom image preprocessing pipeline is:

Convert to grayscale
Resize to 28 × 28
Convert to tensor
Normalize

<img width="493" height="171" alt="image" src="https://github.com/user-attachments/assets/7713741e-2eac-4899-84f3-8348bce1dad1" />

## 4️⃣ Train / Validation Split

The original Fashion-MNIST training dataset is divided into:

80% Training
20% Validation

<img width="271" height="86" alt="image" src="https://github.com/user-attachments/assets/6337cd4a-1fca-4346-8084-965000cc5224" />

