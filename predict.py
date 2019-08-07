import cv2
import numpy as np
from PIL import Image
from keras.models import load_model


def convert_to_array(img):
    im = cv2.imread(img)
    img_ = Image.fromarray(im, 'RGB')
    image = img_.resize((50, 50))
    return np.array(image)


def get_cell_name(label):
    if label == 0:
        return "Infected"
    if label == 1:
        return "Uninfected"


def predict_cell(file):
    model = load_model('./model/cells.h5')
    print("Predicting the type of cell.")
    ar = convert_to_array(file)
    ar = ar / 255
    a = [ar]
    a = np.array(a)
    score = model.predict(a, verbose=1)
    print(score)
    label_index = np.argmax(score)
    print(label_index)
    cell = get_cell_name(label_index)
    return "The predicted cell is a " + cell + " cell."


print(predict_cell('test_cell_images/test.png'))
