# prepare_catdog_dataset.py
import os
import shutil
import random

# --- Configuration ---
SOURCE_KAGGLE_TRAIN_DIR = "path/to/your/downloaded_kaggle_unzipped/train" # !!! UPDATE THIS PATH !!!
BASE_PROJECT_DATA_DIR = "vision_classification_project/data/cats_vs_dogs_subset" # Where the new structured dataset will go

TRAIN_DIR = os.path.join(BASE_PROJECT_DATA_DIR, "train")
VALIDATION_DIR = os.path.join(BASE_PROJECT_DATA_DIR, "validation")

# Number of images per class for train and validation
# Adjust these numbers as needed. For a quick test, keep them small.
# For more robust training, increase them (e.g., 9000 train, 1000 validation)
# Max per class from Kaggle train set is 12500
NUM_TRAIN_PER_CLASS = 1000
NUM_VALID_PER_CLASS = 250

# Create directories
os.makedirs(os.path.join(TRAIN_DIR, "cats"), exist_ok=True)
os.makedirs(os.path.join(TRAIN_DIR, "dogs"), exist_ok=True)
os.makedirs(os.path.join(VALIDATION_DIR, "cats"), exist_ok=True)
os.makedirs(os.path.join(VALIDATION_DIR, "dogs"), exist_ok=True)

def copy_files(filenames, destination_folder):
    for fname in filenames:
        src_path = os.path.join(SOURCE_KAGGLE_TRAIN_DIR, fname)
        dst_path = os.path.join(destination_folder, fname)
        try:
            shutil.copyfile(src_path, dst_path)
        except FileNotFoundError:
            print(f"Warning: Source file {src_path} not found. Skipping.")
        except Exception as e:
            print(f"Error copying {src_path} to {dst_path}: {e}")


# Get all cat and dog filenames from the Kaggle train directory
all_filenames = os.listdir(SOURCE_KAGGLE_TRAIN_DIR)
cat_filenames = [fname for fname in all_filenames if fname.startswith('cat.') and fname.endswith('.jpg')]
dog_filenames = [fname for fname in all_filenames if fname.startswith('dog.') and fname.endswith('.jpg')]

# Shuffle them to get a random subset (optional but good)
random.shuffle(cat_filenames)
random.shuffle(dog_filenames)

print(f"Found {len(cat_filenames)} cat images and {len(dog_filenames)} dog images in source.")

if len(cat_filenames) < (NUM_TRAIN_PER_CLASS + NUM_VALID_PER_CLASS) or \
   len(dog_filenames) < (NUM_TRAIN_PER_CLASS + NUM_VALID_PER_CLASS):
    print("Error: Not enough source images to create the desired train/validation split.")
    print(f"Requested {NUM_TRAIN_PER_CLASS + NUM_VALID_PER_CLASS} per class.")
    exit()

# --- Cats ---
# Training cats
train_cats_to_copy = cat_filenames[:NUM_TRAIN_PER_CLASS]
copy_files(train_cats_to_copy, os.path.join(TRAIN_DIR, "cats"))
print(f"Copied {len(train_cats_to_copy)} cat images to training set.")

# Validation cats
validation_cats_to_copy = cat_filenames[NUM_TRAIN_PER_CLASS : NUM_TRAIN_PER_CLASS + NUM_VALID_PER_CLASS]
copy_files(validation_cats_to_copy, os.path.join(VALIDATION_DIR, "cats"))
print(f"Copied {len(validation_cats_to_copy)} cat images to validation set.")

# --- Dogs ---
# Training dogs
train_dogs_to_copy = dog_filenames[:NUM_TRAIN_PER_CLASS]
copy_files(train_dogs_to_copy, os.path.join(TRAIN_DIR, "dogs"))
print(f"Copied {len(train_dogs_to_copy)} dog images to training set.")

# Validation dogs
validation_dogs_to_copy = dog_filenames[NUM_TRAIN_PER_CLASS : NUM_TRAIN_PER_CLASS + NUM_VALID_PER_CLASS]
copy_files(validation_dogs_to_copy, os.path.join(VALIDATION_DIR, "dogs"))
print(f"Copied {len(validation_dogs_to_copy)} dog images to validation set.")

print("Dataset preparation complete!")
print(f"Training data in: {TRAIN_DIR}")
print(f"Validation data in: {VALIDATION_DIR}")