from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
import os

DataDir = r"C:\Users\shash\PycharmProjects\car_damage\data1a"

train_dir = os.path.join(DataDir, 'training/')
val_dir = os.path.join(DataDir, 'validation/')
train_damage = os.path.join(train_dir, '00-damage')
train_not_damage = os.path.join(val_dir, '01-whole')

num_train_damage = len(os.listdir(train_damage))
num_train_not_damage = len(os.listdir(train_not_damage))

val_damage = os.path.join(val_dir, '00-damage')
val_not_damage = os.path.join(val_dir, '01-whole')

num_val_damage = len(os.listdir(val_damage))
num_val_not_damage = len(os.listdir(val_not_damage))
num_train = num_train_damage + num_train_not_damage
num_val = num_val_damage + num_val_not_damage

total_images = num_val + num_train
print("Total training images", num_train)
print("Total training images (Damaged)", num_train_damage)
print("Total training images (Damaged)", num_train_not_damage)
print()
print("Total validation images", num_val)
print("Total training images (Damaged)", num_val_damage)
print("Total training images (Damaged)", num_val_not_damage)
print()
print("Total Number of Images: ", total_images)

plt.grid('')
image = plt.imread(r"C:\Users\shash\PycharmProjects\car_damage\data1a\training\01-whole\0195.jpg")
plt.imshow(image)
plt.show()

initial_lr = 0.001
epochs = 100
batch_size = 64
classes = ["00-damage", "01-whole"]

print("[INFO] loading images...")

data = []
labels = []

for class_ in classes:
    path = os.path.join(train_dir, class_)
    for image in os.listdir(path):
        image_path = os.path.join(path, image)
        image_ = load_img(image_path, target_size=(224, 224))
        image_ = img_to_array(image_)
        image_ = preprocess_input(image_)

        data.append(image_)
        labels.append(class_)

for class_ in classes:
    path = os.path.join(val_dir, class_)
    for image in os.listdir(path):
        image_path = os.path.join(path, image)
        image_ = load_img(image_path, target_size=(224, 224))
        image_ = img_to_array(image_)
        image_ = preprocess_input(image_)

        data.append(image_)
        labels.append(class_)

lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

data = np.array(data, dtype="float32")
labels = np.array(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.20, stratify=labels, random_state=42)

aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest")

model_base = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))
model_head = model_base.output
model_head = MaxPooling2D(pool_size=(5, 5))(model_head)
model_head = Flatten(name="flatten")(model_head)
model_head = Dense(128, activation="relu")(model_head)
model_head = Dropout(0.5)(model_head)
model_head = Dense(2, activation="softmax")(model_head)
model_final = Model(inputs=model_base.input, outputs=model_head)

for layer in model_base.layers:
    layer.trainable = False
optim = Adam(lr=initial_lr, decay=initial_lr / epochs)
model_final.compile(loss="binary_crossentropy", optimizer=optim, metrics=["accuracy"])

model_train = model_final.fit(
    aug.flow(trainX, trainY, batch_size=batch_size),
    steps_per_epoch=len(trainX) // batch_size,
    validation_data=(testX, testY),
    validation_steps=len(testX) // batch_size,
    epochs=epochs)

predict = model_final.predict(testX, batch_size=batch_size)
predict_index = np.argmax(predict, axis=1)
print(classification_report(testY.argmax(axis=1), predict_index, target_names=lb.classes_))

model_final.save("Car_detection.model", save_format="h5")
