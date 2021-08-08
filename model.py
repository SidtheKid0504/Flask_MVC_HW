import numpy as np
from numpy.core.fromnumeric import clip
import pandas as pd
import PIL.ImageOps as pio
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score
from PIL import Image

y = pd.read_csv('https://raw.githubusercontent.com/whitehatjr/datasets/master/C%20122-123/labels.csv')['labels']
X = np.load('image (1).npz')['arr_0']

print(pd.Series(y).value_counts())

classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
n_classes = len(classes)

train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=9, train_size=10725, test_size=2500)
train_X_scaled = train_X / 255
test_X_scaled = test_X / 255

lr = LogisticRegression(solver='saga', multi_class='multinomial')

lr.fit(train_X_scaled, train_y)

y_preds = lr.predict(test_X_scaled)
acc = accuracy_score(test_y, y_preds)
print(str(acc*100) + '%')

def predict_input_image(img):

    conv_img = Image.open(img).convert('L')
    resized_img = conv_img.resize((28, 28), Image.ANTIALIAS)

    pixel_filter = 20
    min_pixel = np.percentile(resized_img, pixel_filter)
    max_pixel = np.max(resized_img)

    scaled_img = np.clip(resized_img - min_pixel, 0, 255)
    scaled_img = np.asarray(scaled_img) / max_pixel

    test_sample = np.asarray(scaled_img).reshape(1, 784)
    test_pred = lr.predict(test_sample)

    return test_pred[0]