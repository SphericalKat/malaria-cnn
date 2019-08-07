# Importing Necessary Libraries.
import os

import cv2
import keras
import numpy as np  # linear algebra
import sklearn
from PIL import Image
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.models import Sequential
from sklearn.model_selection import train_test_split

data = []
labels = []

# Load infected images
Parasitized = os.listdir("./train_cell_images/Parasitized/")
for p in Parasitized:
    try:
        image = cv2.imread("./train_cell_images/Parasitized/" + p)
        image_from_array = Image.fromarray(image, 'RGB')
        size_image = image_from_array.resize((50, 50))
        rotated45 = size_image.rotate(45)
        rotated75 = size_image.rotate(75)
        blur = cv2.blur(np.array(size_image), (10, 10))
        data.append(np.array(size_image))
        data.append(np.array(rotated45))
        data.append(np.array(rotated75))
        data.append(np.array(blur))
        labels.append(0)
        labels.append(0)
        labels.append(0)
        labels.append(0)
    except AttributeError:
        pass

# Load uninfected images
Uninfected = os.listdir("./train_cell_images/Uninfected/")
for u in Uninfected:
    try:
        image = cv2.imread("./train_cell_images/Uninfected/" + u)
        image_from_array = Image.fromarray(image, 'RGB')
        size_image = image_from_array.resize((50, 50))
        rotated45 = size_image.rotate(45)
        rotated75 = size_image.rotate(75)
        data.append(np.array(size_image))
        data.append(np.array(rotated45))
        data.append(np.array(rotated75))
        labels.append(1)
        labels.append(1)
        labels.append(1)
    except AttributeError:
        pass

# Convert image pixels to numpy arrays for easy processing
cells = np.array(data)
labels = np.array(labels)

# np.save("model/cells", cells)
# np.save("model/labels", labels)
#
# cells = np.load("model/cells.npy")
# labels = np.load("model/labels.npy")

# Shuffle cells to prevent some sort of bias
s = np.arange(cells.shape[0])
np.random.shuffle(s)
cells = cells[s]
labels = labels[s]

num_classes = len(np.unique(labels))
len_data = len(cells)

# Split into train and test datasets
x_train, x_test, y_train, y_test = train_test_split(cells, labels)

x_train = x_train.astype('float32') / 255  # Normalize RGB values by dividing with 255
x_test = x_test.astype('float32') / 255
train_len = len(x_train)
test_len = len(x_test)

# Convert to one-hot encoding as classifier has binary classes
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

print(y_test.shape)
# Create a sequential keras model
model = Sequential()
model.add(Conv2D(filters=16, kernel_size=3, padding="same", activation="relu", input_shape=(50, 50, 3)))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=32, kernel_size=3, padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=64, kernel_size=3, padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(1000, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(500, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(2, activation="softmax"))  # 2 represent output layer neurons
model.summary()
# compile the model with loss function as binary_crossentropy and using adam optimizer you can test result by trying
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit (train) the model. Using a batch size which is x^2 optimizes training on my GPU
model.fit(x_train, y_train, batch_size=512, epochs=20, verbose=1)

accuracy = model.evaluate(x_test, y_test, verbose=1)
print('\n', 'Test_Accuracy: ', accuracy[1])

model.save('./model/cells.h5')
