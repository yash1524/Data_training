import os
import numpy as np
from tensorflow.keras.utils import to_categorical

from keras.layers import Input, Dense
from keras.models import Model
from tensorflow.keras.optimizers import RMSprop

is_init = False
size = -1

class_labels = []  # List to store unique class labels
label_to_index = {}  # Dictionary mapping labels to indices

# Load data and create labels
for i in os.listdir("/home/yash/Desktop/DE project/liveEmoji"):  # Assuming your data is here
    if i.split(".")[-1] == "npy" and i != "labels.npy":  # Exclude "labels.npy"
        if not is_init:
            is_init = True
            X = np.load(i)
            size = X.shape[0]
            y = np.array([i.split('.')[0]] * size).reshape(-1, 1)
        else:
            X = np.concatenate((X, np.load(i)))
            y = np.concatenate((y, np.array([i.split('.')[0]] * size).reshape(-1, 1)))

        class_label = i.split('.')[0]
        if class_label not in class_labels:
            class_labels.append(class_label)
            label_to_index[class_label] = len(class_labels) - 1
        y[:, 0] = label_to_index[class_label]

y = np.array(y, dtype="int32")

# One-hot encode labels
y = to_categorical(y)

# Optional data augmentation (add data augmentation techniques if needed)
# ... (implement data augmentation techniques here using libraries like imgaug)

# Shuffle data
X_new = X.copy()
y_new = y.copy()
counter = 0

cnt = np.arange(X.shape[0])
np.random.shuffle(cnt)

for i in cnt:
    X_new[counter] = X[i]
    y_new[counter] = y[i]
    counter += 1

# Define the model architecture
num_features = X.shape[1]  # Get the number of features from X.shape
num_classes = len(class_labels)  # Number of classes based on unique labels

ip = Input(shape=(num_features,))

m = Dense(512, activation="relu")(ip)
m = Dense(256, activation="relu")(m)

op = Dense(num_classes, activation="softmax")(m)  # Output layer with num_classes units

model = Model(inputs=ip, outputs=op)

# Compile the model
model.compile(optimizer=RMSprop(), loss="categorical_crossentropy", metrics=['acc'])  # Using TensorFlow 2.x API

# Train the model
model.fit(X_new, y_new, epochs=50)

# Save the model and labels
model.save("model.h5")
np.save("labels.npy", np.array(class_labels))
