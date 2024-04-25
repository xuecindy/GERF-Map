#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os

def list_directory_contents(directory):
    try:
        with os.scandir(directory) as entries:
            for entry in entries:
                print(entry.name)
    except FileNotFoundError:
        print("Directory not found.")

# Replace 'directory_path' with the path to the directory you want to list
directory_path = '//home/wenyue.xue/HearingLoss'
list_directory_contents(directory_path)


# In[ ]:


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


# In[ ]:


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
        data_normalized = (data - ave_val) / (max_val - min_val)
        #data_normalized = (data - min_val) / (max_val - min_val)
    
    return data_normalized


# In[ ]:


def load_data_and_labels(folder_path):
    X = []  # To store data
    Y = []  # To store labels
    Z = []  # To store sample name

    for filename in os.listdir(folder_path):
        #print(filename)
        file_path = os.path.join(folder_path, filename)

        # Check if it's a file and not a directory
        if os.path.isfile(file_path):
            # Process the file (you'll need to replace this with your actual data processing)
            data = process_file(file_path)  # Implement this function based on your data format
            #data = np.loadtxt(file_path, delimiter=None)
            X.append(data)

            # Assign label based on file name
            Y.append(0 if filename.startswith('Control') else 1)

            # Assign file name to corresponding sample
            Z.append(filename)

    return np.array(X), np.array(Y), Z


# In[ ]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.initializers import RandomNormal, HeNormal

def create_cnn_model(input_shape, num_classes, learning_rate=0.001):
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape, kernel_initializer=HeNormal(),name='conv1'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu', kernel_initializer=HeNormal(),name='conv2'),
        MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu', name='conv3'),
        Flatten(),
        Dense(128, activation='relu', kernel_initializer=HeNormal()),
        Dense(num_classes, activation='softmax', kernel_initializer=HeNormal())
    ])
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# In[ ]:


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


# In[ ]:


import matplotlib.pyplot as plt
import cv2
import numpy as np

def display_heatmap_only(img, heatmap, alpha=0.4, size=(36, 21), save_path=None):
    # Resize heatmap for visualization
    heatmap_resized = cv2.resize(heatmap, size)
    
    # Convert heatmap to RGB
    heatmap_resized = np.uint8(255 * heatmap_resized)
    heatmap_resized = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    
    # Create a figure to display the results
    plt.figure(figsize=(6, 6))
    
    # Display Heatmap with Color Bar
    im = plt.imshow(heatmap_resized, cmap='jet')
    plt.axis('on')  # Show or hide the axis as per your requirement
    plt.title('Heatmap')
    
    # Add color bar
    plt.colorbar(im, fraction=0.046, pad=0.04, label='Importance')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or show the plot
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        print(f"Image saved to {save_path}")
    else:
        plt.show()


# In[ ]:


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


# Function to plot average training and validation loss and accuracy for all 20 epoch situation
def plot_avg_training_validation_loss_accuracy(all_folds_history, save_path=None, font_size=16, line_width=3):
    avg_loss = np.mean([fold['loss'] for fold in all_folds_history], axis=0)
    avg_val_loss = np.mean([fold['val_loss'] for fold in all_folds_history], axis=0)
    avg_accuracy = np.mean([fold['accuracy'] for fold in all_folds_history], axis=0)
    avg_val_accuracy = np.mean([fold['val_accuracy'] for fold in all_folds_history], axis=0)

    epochs = range(1, len(avg_loss) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, avg_loss, 'bo-', label='Average Training Loss', linewidth=line_width)
    plt.plot(epochs, avg_val_loss, 'ro-', label='Average Validation Loss', linewidth=line_width)
    plt.title('Training and Validation Loss', fontsize=font_size)
    plt.xlabel('Epochs', fontsize=font_size)
    plt.ylabel('Loss', fontsize=font_size)
    plt.legend(fontsize=font_size)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, avg_accuracy, 'bo-', label='Average Training Accuracy', linewidth=line_width)
    plt.plot(epochs, avg_val_accuracy, 'ro-', label='Average Validation Accuracy', linewidth=line_width)
    plt.title('Training and Validation Accuracy', fontsize=font_size)
    plt.xlabel('Epochs', fontsize=font_size)
    plt.ylabel('Accuracy', fontsize=font_size)
    plt.legend(fontsize=font_size)

    plt.tight_layout()

    # Save to file if save_path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


# In[ ]:


from sklearn.model_selection import KFold
from scipy.ndimage import median_filter
from tensorflow.keras.utils import to_categorical
import numpy as np
import pandas as pd
import cv2

#Loop Through Each Subfolder
base_folder = '//home/wenyue.xue/HearingLossTime_CSV/'  # The folder where your T=0, T=1, ..., T=99 folders are located
Save_folder = '//home/wenyue.xue/HearingLossEachTime/'
average_accuracies = {}  # Keys are T{i}, values are the corresponding average accuracies

for i in range(0,100):  # Assuming subfolders are named 'T=0' through 'T=99'
    subfolder_name = f'T={i}'
    print(f'T={i}')
    subfolder_path = os.path.join(base_folder, subfolder_name)
    save_path = os.path.join(Save_folder,  f'T={i}learning.png')
    X,Y,Z =load_data_and_labels(subfolder_path)
    X_reshaped = X[:, :, :, np.newaxis]
    X = median_filter(X, size=3)
    #X = gaussian_filter(X, sigma=1)

    k = 5  # Number of folds
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    accuracy_scores_k = []
    all_folds_history = []
    #print(len(accuracy_scores_k))
    
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        Y_test_encoded = to_categorical(Y_test, num_classes=2)
        
        # Train the model
        val_split_index = int(0.8 * len(X_train))
        X_train, X_val = X_train[:val_split_index], X_train[val_split_index:]
        Y_train, Y_val = Y_train[:val_split_index], Y_train[val_split_index:]
        
        Y_train_encoded = to_categorical(Y_train, num_classes=2)
        Y_val_encoded = to_categorical(Y_val, num_classes=2)
        # Hyperparameter tuning for learning rate
        model = create_cnn_model(X_reshaped[0].shape, num_classes=2, learning_rate=0.001)  # Recreate model to reset weights

        # Train the model with validation data
        history = model.fit(X_train, Y_train_encoded, validation_data=(X_val, Y_val_encoded),
                    epochs=20, batch_size=12, verbose=0)
        all_folds_history.append(history.history) 
        scores_k = model.evaluate(X_test, Y_test_encoded, verbose=0)
        accuracy_scores_k.append(scores_k[1])
       
    # Calculate the average accuracy
    average_accuracy_k = np.mean(accuracy_scores_k)
    average_accuracies[f'T{i}'] = average_accuracy_k
    print(f'Average Accuracy from {k}-Fold CV for T{i}: {average_accuracy_k:.2f}')
    
    plot_avg_training_validation_loss_accuracy(all_folds_history, save_path=save_path, font_size=16, line_width=3)

    if(average_accuracy_k>0.8):
     # print(subfolder_name)
      model =model
      # Name of the last convolutional layer
      last_conv_layer_name = "conv1"
      # Select the example and add the batch dimension
      for j in range(len(Z)):
        save_path = os.path.join(Save_folder, Z[j]+'.png')
        img_array = np.expand_dims(X[j], axis=0)  # This changes the shape from (36, 21, 1) to (1, 36, 21, 1)
      # Generate the heatmap
        heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
      # Prepare the image for display_gradcam function
        img_to_display = img_array[0].squeeze()
      #print(img_to_display.shape)
        if len(img_to_display.shape) == 2:  # If the image is 2D, convert it to 3D
            img_to_display = np.repeat(img_to_display[..., np.newaxis], 3, axis=2)

        display_heatmap_only(img_to_display, heatmap, alpha=0.4, size=(21, 36), save_path=save_path)


# In[ ]:


from sklearn.model_selection import KFold
from scipy.ndimage import median_filter
from tensorflow.keras.utils import to_categorical
import numpy as np
import pandas as pd
import cv2

#Loop Through Each Subfolder
base_folder = '//home/wenyue.xue/HearingLossTime_CSV/'  # The folder where your T=0, T=1, ..., T=99 folders are located
Save_folder = '//home/wenyue.xue/HearingLossEachTime/'
average_accuracies = {}  # Keys are T{i}, values are the corresponding average accuracies

for i in range(0,100):  # Assuming subfolders are named 'T=0' through 'T=99'
    subfolder_name = f'T={i}'
    print(f'T={i}')
    subfolder_path = os.path.join(base_folder, subfolder_name)
    X,Y,Z =load_data_and_labels(subfolder_path)
    X_reshaped = X[:, :, :, np.newaxis]
    X = median_filter(X, size=3)
    #X = gaussian_filter(X, sigma=1)

    k = 5  # Number of folds
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    accuracy_scores_k = []
    #print(len(accuracy_scores_k))
    
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        Y_test_encoded = to_categorical(Y_test, num_classes=2)
    
        Y_train_encoded = to_categorical(Y_train, num_classes=2)
        # Hyperparameter tuning for learning rate
        model = create_cnn_model(X_reshaped[0].shape, num_classes=2, learning_rate=0.001)  # Recreate model to reset weights
        # Train the model with validation data
        model.fit(X_train, Y_train_encoded, epochs=20, batch_size=12, verbose=0)
        scores_k = model.evaluate(X_test, Y_test_encoded, verbose=0)
        accuracy_scores_k.append(scores_k[1])
       
    # Calculate the average accuracy
    average_accuracy_k = np.mean(accuracy_scores_k)
    average_accuracies[f'T{i}'] = average_accuracy_k
    print(f'Average Accuracy from {k}-Fold CV for T{i}: {average_accuracy_k:.2f}')
 


# In[ ]:


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


# In[ ]:


print(len(average_accuracies))


# In[ ]:


import matplotlib.pyplot as plt

labels = list(average_accuracies.keys())
accuracies = list(average_accuracies.values())

# Create a new list for sparse x-axis labels
sparse_labels = [label if i % 5 == 0 else '' for i, label in enumerate(labels)]

plt.figure(figsize=(14, 6))  # Adjusted figure size for better readability
plt.bar(labels, accuracies, color='skyblue')
plt.xlabel('File Identifier')
plt.ylabel('Average Accuracy')
plt.title('Average Accuracy for Each Time Point')
plt.xticks(labels, sparse_labels, rotation=0)  # Use sparse_labels here
plt.ylim(0,1)
plt.grid(True) 
plt.tight_layout()  # Use tight layout to ensure everything fits without overlapping
plt.show()


# In[ ]:




