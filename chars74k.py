import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

# Step 1: Set dataset path
dataset_path = "C:/Users/KIIT/Downloads/archive (1)/EnglishFnt/English/Fnt"  # Replace with your dataset path
output_train_csv = "chars74k_train.csv"
output_test_csv = "chars74k_test.csv"

# Step 2: Initialize data storage
data = []
labels = []

# Step 3: Read images and labels
for root, _, files in os.walk(dataset_path):
    for file in files:
        if file.endswith(('.png', '.jpg', '.jpeg', '.bmp')):  # Supported image formats
            label = os.path.basename(root)  # Folder name as label
            file_path = os.path.join(root, file)
            
            # Load image and convert to grayscale
            image = Image.open(file_path).convert('L')  # Convert to grayscale
            
            # Resize to a fixed size (e.g., 28x28 like MNIST)
            image = image.resize((28, 28))
            
            # Convert image to numpy array
            image_array = np.array(image).flatten()  # Flatten into a 1D array
            
            # Append to data and labels
            data.append(image_array)
            labels.append(label)

# Step 4: Convert to numpy arrays
data = np.array(data)
labels = np.array(labels)

# Step 5: Split data into train and test sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42, stratify=labels)

# Step 6: Create DataFrames for train and test sets
train_df = pd.DataFrame(X_train)
train_df.insert(0, "label", y_train)  # Add labels as the first column

test_df = pd.DataFrame(X_test)
test_df.insert(0, "label", y_test)  # Add labels as the first column

# Step 7: Save to CSV files
train_df.to_csv(output_train_csv, index=False)
test_df.to_csv(output_test_csv, index=False)

print(f"Training data saved to {output_train_csv}")
print(f"Testing data saved to {output_test_csv}")
