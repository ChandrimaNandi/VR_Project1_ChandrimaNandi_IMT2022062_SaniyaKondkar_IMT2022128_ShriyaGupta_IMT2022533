# Mask Detection CNN with PyTorch 🦠🤖

This project implements an image classification model using a Convolutional Neural Network (CNN) built with PyTorch. The model is designed to classify images into two categories: "with_mask" and "without_mask." The purpose is to build an efficient mask detection system using deep learning techniques to help ensure public safety during health crises.

## 📋 Overview

The goal of this project is to create a CNN model capable of accurately detecting whether a person is wearing a mask or not in an image. By leveraging a dataset of masked and unmasked images, the model uses various image transformations for preprocessing and augmentation, followed by training using the Adam optimizer and Cross-Entropy Loss. The model is evaluated on a test set, and early stopping is applied during training to avoid overfitting.

## 🛠 Dependencies

To run this project, the following libraries are required:

- `torch` (for building and training the model)
- `torchvision` (for data transformation and augmentation)
- `numpy` (for handling numerical data)
- `matplotlib` (for visualizations, optional)
- `PIL` (for image processing)

You can install the necessary dependencies using pip:

```bash
pip install torch torchvision numpy matplotlib pillow
```

## 📂 Dataset

The dataset is expected to have the following directory structure:

```
dataset/
    with_mask/
        image1.jpg
        image2.jpg
        ...
    without_mask/
        image1.jpg
        image2.jpg
        ...
```

### Data Preprocessing and Augmentation:
- **Resizing**: Images are resized to a fixed size (e.g., 128x128 pixels).
- **Normalization**: Image pixel values are normalized to a range between 0 and 1.
- **Augmentation**: Random horizontal flips and rotations are applied to increase the robustness of the model and prevent overfitting.

The dataset is split into training and validation sets for proper model evaluation.

## 🏗 Model Architecture (MaskCNN)

The MaskCNN architecture consists of the following layers:

1. **Convolutional Layers**: These layers learn spatial hierarchies of features from the image data. They use filters to convolve over the image to extract low-level features.
2. **Max Pooling Layers**: Pooling layers reduce the spatial dimensions of the feature maps, making the model computationally efficient and improving its generalization.
3. **Fully Connected Layers**: These layers perform the final classification, using the features extracted by the convolutional layers.
4. **Activation Functions**: ReLU is used in convolutional layers to introduce non-linearity, while the final output layer uses a sigmoid activation function to classify the images into two categories.

### Architecture Summary:
- **Conv Layer 1**: 32 filters, 3x3 kernel
- **Conv Layer 2**: 64 filters, 3x3 kernel
- **Fully Connected Layer**: 128 units
- **Output Layer**: 1 unit with sigmoid activation

## 🏋️‍ Training Process

### Hyperparameters:
- **Optimizer**: Adam optimizer, known for its efficient handling of sparse gradients.
- **Loss Function**: Cross-Entropy Loss, suitable for binary classification tasks.
- **Epochs**: The model trains for a pre-defined number of epochs, with early stopping based on validation loss to prevent overfitting.
- **Batch Size**: A batch size of 32 images is used for training.

### Early Stopping:
The model utilizes early stopping to halt training if the validation loss stops improving for a set number of epochs, ensuring efficient training and preventing overfitting.

The best model is saved based on the lowest validation loss observed during training.

## 🔍 Evaluation

Once the model is trained, its performance is evaluated on the test set. The accuracy of the model on the test set is **95.67%**, indicating that it performs well in distinguishing between images of people with and without masks.

## 🚀 Usage

### Setting up the environment:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/mask-detection-cnn.git
   cd mask-detection-cnn
   ```

2. **Install dependencies**:
   Make sure to install the necessary Python libraries using the command:
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare the dataset**:
   Download or organize your dataset with the appropriate folder structure (`with_mask` and `without_mask`). You can also use your custom dataset.

4. **Train the model**:
   Run the training script:
   ```bash
   python train.py --data_dir /path/to/dataset
   ```

   The model will be trained and the best performing model will be saved to a file like `mask_cnn_best.pth`.

5. **Test the model**:
   To evaluate the model on a test set:
   ```bash
   python test.py --model_path /path/to/mask_cnn_best.pth --test_dir /path/to/test_data
   ```

## 📊 Results

- **Test Accuracy**: 95.67%
  The model achieves a high test accuracy, demonstrating its ability to effectively classify masked and unmasked images.

## 🚀 Potential Improvements

- **Hyperparameter Tuning**: Further experimentation with different learning rates, batch sizes, or optimizer settings could improve model performance.
- **Advanced Architectures**: Explore deeper CNN architectures, such as ResNet or VGG, to further improve accuracy.
- **Transfer Learning**: Fine-tuning a pre-trained model (e.g., using a pre-trained ResNet or MobileNet) could accelerate training and improve performance.
- **More Data Augmentation**: Explore additional augmentation techniques, such as random brightness/contrast adjustments or color jitter, to make the model more robust.

## 🤖 Conclusion

This mask detection CNN model is a powerful tool for automatic mask detection in images. With a high accuracy of 95.67%, it provides an effective solution for ensuring compliance with health guidelines in public spaces. Future improvements could enhance its generalization and performance even further!

---

Feel free to modify or enhance the code, experiment with different architectures, or integrate it into a real-time application for mask detection! 😷
