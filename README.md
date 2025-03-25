# Part A

## Introduction

This part focuses on classifying face images into two categories: "with mask" and "without mask." Various feature extraction techniques and machine learning models are used to achieve this classification, with a focus on handcrafted features such as HOG, LBP, and edge detection.

## Dataset

The dataset consists of images with mask and without mask :
- FaceMaskDetection_dataset/with_mask : images with mask
- FaceMaskDetection_dataset/without_mask : images without mask

Images are loaded from designated folders, resized, and labeled accordingly. The preprocessed images are then used for feature extraction and classification.

## Methodology

1. **Data Preprocessing**
- **Resizing:** All images are resized to **(64x64)** for uniformity and computational efficiency.
- **Grayscale Conversion:** Reduces color-related noise and simplifies feature extraction.
- **Histogram Equalization:** Enhances contrast by redistributing pixel intensity values, improving the visibility of facial features.

2. **Feature Extraction Approaches**

The following feature extraction methods were applied:
- **Histogram of Oriented Gradients (HOG):** 
  - **Orientations:** 9 (captures fine details without excessive noise).
  - **Pixels per cell:** (8,8) (balances detail capture and computational efficiency).
  - **Cells per block:** (2,2) (ensures robust feature normalization).
  - **Block norm:** L2-Hys (improves illumination invariance).
  - **Performance:** HOG proved to be the most effective extractor, capturing fine-grained details and spatial edge information.

- **Local Binary Patterns (LBP):**
  - Captures local texture variations through pixel neighborhood relationships.
  - Histogram normalization ensures consistency.
  - **Weakness:** LBP lacks spatial edge direction sensitivity, making it less effective than HOG.

- **Canny Edge Detection + HOG:**
  - Canny edge detection highlights major contours.
  - HOG is applied on edge-detected images.
  - **Weakness:** This method enhances prominent edges but loses finer texture details, reducing classification performance.

- **Principal Component Analysis (PCA):**
  - Used for dimensionality reduction.
  - Retains the most significant features while reducing computational complexity.

3. **Model Training**
   
Feature vectors were standardized before model training to ensure consistency across varying feature scales. Two machine learning models were trained:

- **Support Vector Machine (SVM):**
  - **Kernel:** Radial Basis Function (RBF) (handles non-linearity effectively).
  - **Regularization parameter (C):** Optimally chosen for balance between margin maximization and generalization.
  - **Gamma:** Tuned for best performance.
  - **Performance:** Achieved high accuracy due to its ability to find optimal decision boundaries in high-dimensional space.

- **Multi-Layer Perceptron (MLP):**
  - **Hidden Layers:** (256, 128, 64) (optimized for learning efficiency).
  - **Activation Function:** ReLU.
  - **Maximum Iterations:** 1000.
  - **Performance:** Comparable accuracy to SVM but required more training time.

## Results
- **SVM Accuracy:** **0.93** (high accuracy due to effective feature extraction).
- **MLP Accuracy:** **0.93** (similar performance but more computationally intensive).

## Observations
- **HOG emerged as the most effective feature extraction technique** due to its ability to capture fine details and spatial edge structures.
- **SVM performed as well as MLP** but was computationally more efficient.
- **LBP and Canny+HOG were less effective** due to missing crucial spatial information.
- **PCA helped in dimensionality reduction** but was not the best standalone feature extractor.

___

# Part B

## Introduction

This project implements a CNN-based binary classification model to detect whether individuals are wearing masks. The model is trained and evaluated on an image dataset with two classes: "with_mask" and "without_mask."

## Dataset

The dataset is structured in the `data/dataset` directory with the following subdirectories:
- `with_mask/`: Contains images of individuals wearing masks.
- `without_mask/`: Contains images of individuals without masks.

## Methodology

1. **Data Preprocessing**
   
- The dataset was loaded using `torchvision.datasets.ImageFolder`, which automatically assigns labels based on subdirectory names.
- A series of transformations were applied using `torchvision.transforms.Compose`:
  - **Resizing:** All images were resized to 150x150 pixels.
  - **Random Rotation:** Images were rotated randomly by up to 30 degrees.
  - **Random Horizontal Flip:** 50% chance of flipping images to introduce symmetry invariance.
  - **Random Resized Crop:** Cropped images to 150x150 with a scale between 80% and 100%.
  - **Normalization:** Pixel values were normalized to the range [-1, 1].
- The dataset was split into 80% training and 20% testing using `torch.utils.data.random_split`.
- `DataLoader` instances were created with a batch size of 32.

2. **Model Training:(MaskCNN)**
   
The CNN model consists of:
- **Two convolutional layers** with 100 filters, a 3x3 kernel, and padding of 1.
- **Max-pooling layers** (2x2) to reduce spatial dimensions.
- **Two fully connected layers**:
  - First layer: 100×37×37 → 50 features.
  - Second layer: 50 → 2 output classes.
- **ReLU activation** functions after convolutional and first fully connected layers.
- **Loss Function:** CrossEntropyLoss for multi-class classification.
- **Optimizer:** Adam with a learning rate of 0.001.
- **Early Stopping:** Training stopped if validation loss did not improve for 5 epochs (minimum delta = 0.001).
- Training ran for a **maximum of 15 epochs** with the following steps per epoch:
  - Model trained using `model.train()`.
  - Forward pass → loss calculation → backward pass → weight update.
  - Validation loss and accuracy computed using `model.eval()`.

3. **Hyperparameters & Tuning**
- **CNN Architecture:** Two convolutional layers, two max-pooling layers, two fully connected layers.
- **Optimizer:** Adam with a learning rate of 0.001.
- **Loss Function:** Cross-Entropy Loss.
- **Batch Size:** 32 for both training and testing.
- **Number of Epochs:** 15 (early stopping applied).
- **Data Augmentation:** Included resizing, rotation, horizontal flipping, and cropping.

4. **Evaluation:**
   
- The best model (based on validation loss) was saved.
- The model was evaluated on the test dataset:
  - Model predictions were compared with true labels.
  - Test accuracy was computed.

## Result
- The final test accuracy achieved was **95.67%**, indicating a strong generalization ability.

## Observation
- **High test accuracy** suggests the model effectively distinguishes between masked and unmasked individuals.
- **Data augmentation** contributed to improved robustness.
- **Early stopping** helped prevent overfitting.
- **Challenges faced**:
  - Computational cost was addressed using a GPU.
  - Overfitting was mitigated through data augmentation and early stopping.
  - Hyperparameter tuning required experimentation.
- **Future Work**:
  - Exploring deeper architectures.
  - Evaluating on larger, more diverse datasets.
___

# Part C

## Introduction

This section explores the implementation and evaluation of three traditional image segmentation techniques—Gaussian Mixture Model (GMM), Otsu's thresholding, and the Watershed algorithm—on a dataset of face crops. The performance of these methods is compared using the Intersection over Union (IoU) and Dice score metrics.

## Dataset

The dataset consists of images and their corresponding segmentation masks. It is structured as follows:
- `img/`: Original images.
- `face_crop/`: Cropped masked face images.
- `face_crop_segmentation/`: Segmentation masks.

## Methodology

1. **Data Preprocessing**

- Paths to image and ground truth mask folders are defined.
- Image and mask filenames are retrieved using `os.listdir()`.
- For each image-mask pair:
  - The image is loaded using `cv2.imread()`, converted from BGR to RGB using `cv2.cvtColor()`, and handled for potential loading errors.
  - The ground truth mask is loaded as a grayscale image using `cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)`, with error handling.
  - The ground truth mask is converted to a PyTorch tensor (`torch.uint8`).

2. **Model Training:**
   
- The three segmentation methods (`gmm_segmentation`, `otsu_segmentation`, and `watershed_segmentation`) are applied to the images within a `try-except` block to handle potential errors.
- The predicted masks are resized to match the dimensions of the ground truth masks using `cv2.resize()` with `cv2.INTER_NEAREST` interpolation.
- The resized masks are converted back to PyTorch tensors (`torch.uint8`) for evaluation.

3. **Hyperparameters & Tuning**
- **GMM Segmentation:**
  - Number of components: **2** (assumes foreground and background).
  - Covariance type: **"tied"** (shared covariance across components).
  - Random state: **42** (ensures reproducibility).
  
- **Otsu’s Thresholding:**
  - An automatic thresholding technique that dynamically determines the optimal threshold without manual hyperparameters.
  
- **Watershed Algorithm:** 
  - Morphological operations kernel size: **3x3**.
  - Number of iterations for morphological opening: **2**.
  - Number of iterations for dilation: **3**.
  - Distance transform threshold factor: **0.5**.

4. **Evaluation:** 
- The `evaluate_segmentation` function computes IoU and Dice scores by comparing predicted masks with ground truth masks.
- Per-image scores are printed to monitor performance.
- Mean IoU and Dice scores for each method are computed using `np.mean()`.

## Result
The final mean IoU and Dice scores for each segmentation method are:

| Segmentation Method | IoU  | Dice Score |
|---------------------|------|------------|
| **GMM**            | 0.3600 | 0.4949     |
| **Otsu**           | 0.3669 | 0.5037     |
| **Watershed**      | 0.1422 | 0.2013     |

## Observation

**Strengths and Weaknesses of Each Method:** 
- **GMM Segmentation:** 
  - Works well when pixel intensity distributions are distinct.
  - Assumes that the cluster with the highest mean intensity represents the foreground, which may not always be accurate.

- **Otsu’s Thresholding:** 
  - Simple and computationally efficient.
  - Relies on a bimodal intensity histogram; may perform poorly if the face and background are not well separated.

- **Watershed Algorithm:** 
  - Highly dependent on preprocessing and marker accuracy.
  - Sensitive to parameter choices, leading to possible over-segmentation.

### Challenges Faced:
- **Heuristic Assumptions:** 
  - The GMM method assumes that the highest intensity cluster corresponds to the face, which is not always true.
  
- **Parameter Tuning:** 
  - Watershed performance is influenced by morphological parameters, requiring dataset-specific tuning.
  
- **Evaluation Metrics:** 
  - IoU and Dice scores provide numerical comparisons but visual inspection is necessary to understand segmentation quality.

___

# Part D

## Introduction
This project trains a U-Net model for precise mask segmentation and compares its performance with a traditional segmentation method using IoU and Dice scores.

## Dataset
The dataset consists of images and their corresponding segmentation masks. It is structured as follows:
- `img/`: Original images.
- `face_crop/`: Cropped masked face images.
- `face_crop_segmentation/`: Segmentation masks.

## Methodology
1. **Data Preprocessing**
- **Loading & Resizing:** 
  - Images are loaded in **RGB mode** and masks in **grayscale**. 
  - Both are resized to **(256, 256)**.

- **Normalization:** 
  - Images are scaled to **[0,1]**. 
  - Masks are converted to **binary format** (0 for background, 1 for mask).

- **Format Conversion:** 
  - Images are transformed to **(C, H, W)** format. 
  - Masks are reshaped to **(1, H, W)**.

- **Data Augmentation:** 
  - If enabled, **horizontal and vertical flips** are applied to both images and masks.

- **Conversion to Tensors:** 
  - Processed images and masks are converted into **torch tensors** for model training.
  
2. **Model Training:** 
- **Encoder (Downsampling Path)**:
  - 4 convolutional blocks with ReLU activation and batch normalization.
  - Max pooling layers reduce spatial dimensions.
  - Increasing filters: `16 → 32 → 64 → 128`.
  
- **Bottleneck Layer**:
  - Deepest layer with **256** filters and dropout of **0.3**.

- **Decoder (Upsampling Path)**:
  - Transposed convolution layers for upsampling.
  - Skip connections concatenate corresponding encoder features.
  - Decreasing filters: `128 → 64 → 32 → 16`.

- **Output Layer**:
  - **1x1 convolution** to generate a single-channel mask.
  - **Sigmoid activation** to produce pixel-wise probabilities.
  
3. **Hyperparameters & Tuning**
- **Learning Rate:** `5e-4`, reduced dynamically using **ReduceLROnPlateau**.
- **Batch Size:** Tuned based on GPU memory constraints.
- **Epochs:** `50` (Early stopping applied).
- **Dropout:** Applied in encoder and bottleneck layers for regularization.
- **BCE + Dice Loss**:
  - **Binary Cross Entropy (BCE):** Penalizes incorrect pixel classifications.
  - **Dice Loss:** Measures overlap between predicted and ground truth masks.
  - **Formula:** 
    	Loss = BCE + (1 - Dice)
- **Adam Optimizer** (`lr=5e-4`):
  - Adaptive learning rate for faster convergence.
  - Momentum-based updates for stability.

4. **Evaluation:** 
- **IoU (Intersection over Union):**
  - Measures segmentation accuracy.
  - Thresholded at `0.5` for binary masks.

- **Dice Coefficient:**
  - Measures the overlap between prediction and ground truth.
  - Higher values indicate better segmentation performance.

- **Validation Strategy:**
	- **Total dataset size:** `len(image_filenames)`
	- **Test set:** 20% of the total dataset.
	- **Train-validation split:** The remaining 80% is further split into:
	  - **Training set:** 80% of the 80% (i.e., **64% of the total dataset**).
	  - **Validation set:** 20% of the 80% (i.e., **16% of the total dataset**).
	  
## Result
The model achieved a test loss of **0.4749**, a Dice coefficient of **0.5669**, and an IoU score of **0.8331**. The IoU and Dice coefficient plots indicate the model's performance over the epochs, showing a consistent increase in segmentation accuracy. The final IoU score suggests that the model has successfully learned to segment the target regions effectively.

## Observation
From the IoU score plot, we observe that both training and validation IoU steadily improve, stabilizing after approximately 20 epochs. This indicates that the model is learning well and not overfitting. Similarly, the Dice coefficient plot shows a progressive increase, with validation and training scores closely following each other. The slight fluctuations in validation metrics suggest minor variations in the dataset but overall demonstrate stable generalization. The high IoU score of **0.8331** signifies that the predicted masks closely align with the ground truth, making the model effective for segmentation tasks.

___

## How to Run

The following libraries are required for the project:

**Standard Libraries**
- `os` - For file and directory management.
- `cv2` - OpenCV for image processing.
- `numpy` - For numerical computations.
- `pandas` - For handling tabular data.
- `random` - For randomization and seeding.
- `tqdm` - For progress bars.

**Visualization**
- `seaborn` - For enhanced data visualization.
- `matplotlib.pyplot` - For plotting images and graphs.

**Data Processing**
- `sklearn.model_selection.train_test_split` - For splitting datasets.
- `warnings` - To suppress unnecessary warnings.

**Segmentation**
- `torch` - Core PyTorch library.
- `torch.nn` - Neural network module.
- `torch.utils.data.DataLoader, Dataset` - For handling datasets and batching.
- `torch.nn.functional` - Provides activation functions and other utilities.
- `torch.optim` - For optimization algorithms.
- `torchvision.transforms` - Image transformations.

**Image Processing**
- `PIL.Image` - For image loading and manipulation.
- `albumentations` - For data augmentation.
- `segmentation_models_pytorch` - Pretrained segmentation models.

**Filesystem Handling**
- `pathlib.Path` - For handling file paths.

**Installation**
Ensure the required packages are installed before running the code
	


