#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os

def list_directory_contents(directory):
    try:
        with os.scandir(directory) as entries:
            for entry in entries:
                print(entry.name)
    except FileNotFoundError:
        print("Directory not found.")

# Replace 'directory_path' with the path to the directory you want to list
directory_path = '/work/zhoulong/HL/'
list_directory_contents(directory_path)


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import tensorflow as tf
import keras
from keras import layers, models
import io
import os
import cv2


# In[3]:


def process_file(file_path):
    # Load the CSV data into a DataFrame
    df = pd.read_csv(file_path)
    
    # Convert the DataFrame to a NumPy array for numerical operations
    data = df.values  # This converts the entire DataFrame to a NumPy array
    
    # Calculate min, max, and standard deviation
    min_val = np.min(data)
    max_val = np.max(data)
    std_dev = np.std(data)
    ave_val = np.mean(data)
    
    # Check if the standard deviation is zero (i.e., all values are the same)
    if std_dev == 0:
        # Set the data to zero to indicate no variation
        data_normalized = np.zeros(data.shape)
    else:
        # Normalize the data
        data_normalized = (data - min_val) / (max_val - min_val)
        #data_normalized = (data - ave_val) / (max_val - min_val)
    
    return data_normalized


# In[4]:


from scipy.ndimage import median_filter
def load_data_and_labels(folder_path):
    X = []  # To store data
    Y = []  # To store labels
    #Z = []  # To store sample name
    
    for i in range(0, 100):  # Assuming subfolders are named 'T=0' through 'T=99'
        subfolder_name = f'T={i}'
        subfolder_path = os.path.join(folder_path, subfolder_name)
        for filename in os.listdir(subfolder_path):
            file_path = os.path.join(subfolder_path, filename)
            # Check if it's a file and not a directory
            if os.path.isfile(file_path):
                # Process the file
                data = process_file(file_path)  # Implement this function based on your data format
                # Convert data to float32 if it's not already
                data = median_filter(data, size=3)
                data = data.astype(np.float32)          
                X.append(data)
                # Assign label based on file name
                Y.append(0 if filename.startswith('Control') else 1)
                # Assign file name to corresponding sample
                #Z.append(filename)

    return np.array(X, dtype=np.float32), np.array(Y)  # Ensure X is explicitly converted to float32


# In[5]:


X,Y =load_data_and_labels('/work/zhoulong/HL/Time_CSV/')
print(X.shape)
X_reshaped = X[:, :, :, np.newaxis]


# In[7]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.initializers import RandomNormal,HeNormal

def create_cnn_model(input_shape, num_classes, learning_rate=0.001):
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape, kernel_initializer=HeNormal(),name='conv1'),
        MaxPooling2D(pool_size=(2, 2), name='MaxPooling1'),
        Conv2D(64, (3, 3), activation='relu', kernel_initializer=HeNormal(),name='conv2'),
        MaxPooling2D(pool_size=(2, 2), name='MaxPooling2'),
        Conv2D(128, (3, 3), activation='relu', name='conv3'),
        Flatten(),
        Dense(128, activation='relu', kernel_initializer=HeNormal(),  name='Dense'),
        Dense(num_classes, activation='softmax', kernel_initializer=HeNormal(),name='Output')
    ])
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# In[8]:


import matplotlib.pyplot as plt
import tensorflow as tf

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        #print("Output contains NaN:", np.isnan(last_conv_layer_output).any())
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)

    # Handle different dimensions in grads
    if len(grads.shape) == 4:  # Typical case for 4D tensor [batch, height, width, channels]
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    elif len(grads.shape) == 3:  # In case grads is a 3D tensor [height, width, channels]
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
    else:  # Other cases need to be handled specifically
        raise ValueError("Unexpected shape for gradients: " + str(grads.shape))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    epsilon = 1e-10
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + epsilon)
   # heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


# In[9]:



def display_gradcam2(img, heatmap, alpha=0.4, size=(21, 36), save_path=None):
    # Reshape and stretch the heatmap and image for visualization
    heatmap = cv2.resize(heatmap, size)
    img = cv2.resize(img, size)
    # Now, combine and save as done in previous steps

    # Ensure img is in 3-channel RGB
    if img.ndim == 2:
        img = np.stack((img,)*3, axis=-1)
    elif img.shape[2] == 1:  # For grayscale with channel dimension
        img = np.concatenate([img]*3, axis=-1)

    # Convert heatmap to RGB
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    heatmap_float = heatmap.astype(np.float32) / 255.0
    img_float = img.astype(np.float32) / 255.0

    # Superimpose the heatmap on original image
    superimposed_img = heatmap_float * alpha + img_float
    superimposed_img = np.clip(superimposed_img * 255, 0, 255).astype('uint8')

    # Display Grad CAM
    plt.imshow(superimposed_img)
    plt.axis('off')  # Hide the axis

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        print(f"Image saved to {save_path}")
    else:
        plt.colorbar(label='Importance')
        plt.show()


# In[10]:


import numpy as np
import pandas as pd

def save_heatmap_to_csv(heatmap, file_path='heatmap.csv'):
    """
    Save the heatmap data to a CSV file.
    Parameters:
    - heatmap: Numpy array containing the heatmap data.
    - file_path: The local path where the CSV file will be saved.
    """
    # Convert the heatmap numpy array to a DataFrame
    df = pd.DataFrame(heatmap)
    # Save the DataFrame to a CSV file
    df.to_csv(file_path, index=False)
    print(f"Heatmap saved to {file_path}")


# In[11]:


from keras.callbacks import EarlyStopping

# Define the early stopping callback
early_stopping = EarlyStopping(
    monitor='val_loss',  # Monitor the validation loss
    patience=3,         # Number of epochs with no improvement after which training will be stopped
    min_delta=0.001,    # Minimum change in the monitored quantity to qualify as an improvement
    mode='min',         # The direction is automatically inferred if not set, but here 'min' means we want to minimize the loss
    verbose=1           # Print a message when early stopping is triggered
)


# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold
from keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from keras.callbacks import EarlyStopping

# Assuming make_gradcam_heatmap and display_gradcam2 are defined elsewhere
def average_heatmaps(heatmaps):
    """Average a list of heatmaps."""
    return np.mean(heatmaps, axis=0)
# Assuming create_cnn_model, X, and Y are defined

k = 100  # Number of folds
kf = KFold(n_splits=k, shuffle=False)
all_folds_history  = []  # To store history for each fold
accuracy_scores_k = []
heatmaps = []
save_path = '/work/zhoulong/HL/AllTimeHeat/'
#test_accuracy_scores_k = []
for fold_index, (train_index, test_index) in enumerate(kf.split(X), start=1):
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]
    
    Y_test_encoded = to_categorical(Y_test, num_classes=2)

    # Further split X_train for training and validation
    # For simplicity, let's take 10% of each fold's training data as validation data
    val_split_index = int(0.9 * len(X_train))
    X_train, X_val = X_train[:val_split_index], X_train[val_split_index:]
    Y_train, Y_val = Y_train[:val_split_index], Y_train[val_split_index:]
    Y_train_encoded = to_categorical(Y_train, num_classes=2)
    Y_val_encoded = to_categorical(Y_val, num_classes=2)
    # Hyperparameter tuning for learning rate
    model = create_cnn_model(X_reshaped[0].shape, num_classes=2, learning_rate=0.001)  # Recreate model to reset weights

    # Train the model with validation data
    #history = model.fit(X_train, Y_train_encoded, validation_data=(X_val, Y_val_encoded),
                    #epochs=20, batch_size=12, verbose=0,callbacks=[early_stopping])

    history = model.fit(X_train, Y_train_encoded, validation_data=(X_val, Y_val_encoded),
                    epochs=50, batch_size=12, verbose=0)
    all_folds_history.append(history.history) 
    scores_k = model.evaluate(X_test, Y_test_encoded, verbose=0)
    accuracy_scores_k.append(scores_k[1])


# In[ ]:


# Function to plot average training and validation loss and accuracy for all 20 epoch situation
def plot_avg_training_validation_loss_accuracy(all_folds_history):
    avg_loss = np.mean([fold['loss'] for fold in all_folds_history], axis=0)
    avg_val_loss = np.mean([fold['val_loss'] for fold in all_folds_history], axis=0)
    avg_accuracy = np.mean([fold['accuracy'] for fold in all_folds_history], axis=0)
    avg_val_accuracy = np.mean([fold['val_accuracy'] for fold in all_folds_history], axis=0)

    epochs = range(1, len(avg_loss) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, avg_loss, 'bo-', label='Average Training Loss')
    plt.plot(epochs, avg_val_loss, 'ro-', label='Average Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, avg_accuracy, 'bo-', label='Average Training Accuracy')
    plt.plot(epochs, avg_val_accuracy, 'ro-', label='Average Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

plot_avg_training_validation_loss_accuracy(all_folds_history)


# In[ ]:


import matplotlib.pyplot as plt

# Assuming accuracy_scores_k is defined
n = 10  # Display a label for every 10th fold

plt.figure(figsize=(18, 6))  # Specifies the figure size
plt.plot(accuracy_scores_k, marker='o', linestyle='-', color='b')

# Increase font sizes for title and labels
plt.title('Test Accuracy for Each Time Point', fontsize=20)
plt.xlabel('Time Point', fontsize=20)
plt.ylabel('Accuracy', fontsize=20)

# Set x-ticks to be sparse and increase font size for ticks
ticks = [f"T{i}" if i % n == 0 else "" for i in range(len(accuracy_scores_k))]
plt.xticks(range(len(accuracy_scores_k)), ticks, fontsize=16)  # Apply sparse labeling and increase font size

plt.grid(True)  # Adds a grid for easier visualization
plt.show()


# In[ ]:


import matplotlib.pyplot as plt

# Assuming accuracy_scores_k is defined
n = 10  # For sparse x-labels, showing a label for every 10th fold

plt.figure(figsize=(20, 6))  # Specifies the figure size
# Create a bar plot
plt.bar(range(len(accuracy_scores_k)), accuracy_scores_k, color='b')

# Set title and labels with larger font sizes
plt.title('Test Accuracy for Each Fold', fontsize=20)
plt.xlabel('Fold Number', fontsize=16)
plt.ylabel('Accuracy', fontsize=16)

# Set sparse x-ticks with larger font
ticks = [f"Fold {i+1}" if i % n == 0 else "" for i in range(len(accuracy_scores_k))]
plt.xticks(range(len(accuracy_scores_k)), ticks, fontsize=12)  # Apply sparse labeling and set font size

plt.grid(True)  # Adds a grid for easier visualization
plt.show()


# In[ ]:




