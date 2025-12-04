# Transformer Models for Human Reticulocyte Classification

## 📌 Project Overview

This project explores the application of state-of-the-art **Vision Transformer (ViT)** models for the classification of human blood cells. Specifically, it focuses on distinguishing between **Erythrocytes**, **Reticulocytes**, and **Background** noise using the Human Reticulocyte Dataset.

To achieve high performance and computational efficiency, the project implements advanced model optimization techniques, including **Low-Rank Adaptation (LoRA)** and **Unstructured Pruning**. It provides a comprehensive comparison of different transformer architectures under various training scenarios.

## 🚀 Key Features

*   **Multi-Model Architecture**: Implements and compares three distinct transformer models:
    *   **Swin Transformer** (Swin Tiny)
    *   **Vision Transformer** (ViT Base)
    *   **Data-efficient Image Transformer** (DeiT Small)
*   **Model Optimization**:
    *   **LoRA (Low-Rank Adaptation)**: Efficient fine-tuning by injecting trainable rank decomposition matrices into attention layers.
    *   **Pruning**: Applies L1 unstructured pruning to reduce model size and complexity without significant accuracy loss.
*   **Robust Training Pipeline**:
    *   **Data Augmentation**: extensive augmentation techniques (RandomResizedCrop, ColorJitter, RandomErasing, etc.) to improve generalization.
    *   **Dual Scenarios**: Evaluates models both **with** and **without** augmentation to assess robustness.
    *   **Learning Rate Scheduling**: Uses Cosine Annealing for optimal convergence.
    *   **Early Stopping**: Prevents overfitting by monitoring validation accuracy.

## 📂 Dataset

The project utilizes the **Human Reticulocyte Dataset**, organized into three classes:
1.  **Erythrocyte**: Mature red blood cells.
2.  **Reticulocyte**: Immature red blood cells.
3.  **BG (Background)**: Background images or artifacts.

*   **Image Size**: Resized to 224x224 pixels.
*   **Splitting Strategy**: Stratified shuffle split into Train, Validation, and Test sets.

## 🛠️ Models & Configurations

The project evaluates two primary configurations for each model architecture:

| Configuration | Description | Key Techniques |
| :--- | :--- | :--- |
| **Base Model** | Standard fine-tuning | Full parameter update, No pruning |
| **Optimized** | Efficient fine-tuning | **LoRA** adapters injected, **Pruning** applied, Backbone frozen |

## 💻 Installation & Requirements

To run this project, you need Python 3.x and the following libraries:

```bash
pip install torch torchvision torchaudio
pip install timm
pip install pandas numpy scikit-learn matplotlib seaborn opencv-python
```

**Key Dependencies:**
*   `torch`: Deep learning framework.
*   `timm`: PyTorch Image Models library for loading pre-trained transformers.
*   `scikit-learn`: For metrics and data splitting.
*   `matplotlib` & `seaborn`: For visualization.

## 📖 Usage

1.  **Clone the repository** (if applicable) or download the project files.
2.  **Data Setup**: Ensure the dataset is located at the path specified in the `DATA_ROOT` variable within the notebook (default: `/kaggle/input/human-dataset-transformer-models`). You may need to adjust this path to point to your local dataset directory.
3.  **Run the Notebook**: Open `human-dataset-with-transformer-models.ipynb` in Jupyter Notebook, JupyterLab, or VS Code.
4.  **Execute Cells**: Run the cells sequentially to:
    *   Load and preprocess data.
    *   Initialize models with LoRA and Pruning.
    *   Train models under different scenarios.
    *   Evaluate performance and generate comparison tables.

## 📊 Methodology

### 1. Data Preprocessing
*   **Augmentation**: Applies transformations like rotation, flipping, color jitter, and random erasing to the training set in the "With Augmentation" scenario.
*   **Normalization**: Standard ImageNet normalization is applied to all images.

### 2. LoRA Implementation
*   Injects `LoRALinear` layers into the attention mechanisms (`qkv`, `proj`) of the transformer models.
*   Freezes the pre-trained backbone parameters, training only the LoRA adapters and the classification head.

### 3. Pruning
*   Applies conservative L1 unstructured pruning to `Linear` and `Conv2d` layers.
*   Removes a percentage (e.g., 10%) of the weights with the smallest magnitude to compress the model.

## 📈 Results & Performance

The project generates detailed performance metrics, including:
*   **Accuracy, Precision, Recall, F1-Score** for all model variants.
*   **Training Curves**: Visualizes Loss and Accuracy over epochs.
*   **Model Efficiency**: Compares model size (MB), number of trainable parameters, and training time.

*Example findings (based on notebook execution):*
*   **Swin Tiny** often demonstrates excellent performance-to-efficiency balance.
*   **LoRA + Pruning** significantly reduces the number of trainable parameters (e.g., by ~30-40%) while maintaining competitive accuracy, sometimes even outperforming full fine-tuning due to better regularization.

## 📜 License

This project is open-source and available for educational and research purposes.

---
*Created by [Your Name/Organization]*
