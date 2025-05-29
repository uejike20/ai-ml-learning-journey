# Fine-Tuning a Vision Model for Image Classification (Cats vs. Dogs)

This project demonstrates how to fine-tune a pre-trained ResNet18 model for classifying images of cats and dogs using PyTorch.

## Project Structure

## Dataset

This project uses a subset of the "Cats vs. Dogs" dataset. You need to download images of cats and dogs and organize them into the following structure under `vision_classification_project/data/cats_vs_dogs_small/`:
For a quick start, aim for ~100-200 images per class for training and ~20-50 per class for validation.

**Note:** The `data/` and `models/` directories are typically added to `.gitignore` if they contain large files.

## Setup

1.  **Clone the main repository (if you haven't already):**
    ```bash
    # git clone ... your ai-ml-learning-journey repo ...
    # cd ai-ml-learning-journey/vision_classification_project
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Prepare your dataset** as described above and place it in `vision_classification_project/data/`.

## Usage

1.  **Open and run the Jupyter Notebook:**
    Navigate to the `vision_classification_project/` directory in your terminal (with the virtual environment activated).
    ```bash
    jupyter lab
    ```
    Then, open `notebooks/image_classification_fine_tuning.ipynb` in the JupyterLab interface and run the cells.

    The notebook will:
    *   Load and preprocess the image data.
    *   Load a pre-trained ResNet18 model.
    *   Replace the classifier head for the new task (cats vs. dogs).
    *   Fine-tune the model.
    *   Save the best performing model weights to `models/cats_dogs_best_model.pth`.
    *   Plot training and validation loss/accuracy.

## Key Learnings

*   Loading image data using `torchvision.datasets.ImageFolder`.
*   Applying data augmentation and normalization using `torchvision.transforms`.
*   Using pre-trained models from `torchvision.models`.
*   Modifying the classifier layer of a pre-trained model for a new task (transfer learning).
*   Freezing layers of the pre-trained model.
*   Training loop structure in PyTorch.
*   Saving and loading model weights.

## Further Exploration

*   Experiment with different pre-trained models (e.g., ResNet34, VGG16, EfficientNet).
*   Try unfreezing more layers of the pre-trained model for more extensive fine-tuning.
*   Implement more advanced data augmentation techniques.
*   Evaluate the model on a dedicated test set.
*   Build a simple script to predict on new, unseen images.