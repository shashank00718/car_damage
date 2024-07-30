import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import MaxPooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.metrics import classification_report

# Define directories
DataDir = r"C:\Users\shash\PycharmProjects\car_damage\data1a"
train_dir = os.path.join(DataDir, 'training/')
val_dir = os.path.join(DataDir, 'validation/')

# Count images
num_train_damage = len(os.listdir(os.path.join(train_dir, '00-damage')))
num_train_not_damage = len(os.listdir(os.path.join(train_dir, '01-whole')))
num_val_damage = len(os.listdir(os.path.join(val_dir, '00-damage')))
num_val_not_damage = len(os.listdir(os.path.join(val_dir, '01-whole')))
num_train = num_train_damage + num_train_not_damage
num_val = num_val_damage + num_val_not_damage
total_images = num_val + num_train

# Print image counts
print("Total training images", num_train)
print("Total training images (Damaged)", num_train_damage)
print("Total training images (Not Damaged)", num_train_not_damage)
print("Total validation images", num_val)
print("Total validation images (Damaged)", num_val_damage)
print("Total validation images (Not Damaged)", num_val_not_damage)
print("Total Number of Images: ", total_images)

# Display a sample image
plt.grid(False)
image = plt.imread(r"C:\Users\shash\PycharmProjects\car_damage\data1a\training\01-whole\0195.jpg")
plt.imshow(image)
plt.show()

# Set hyperparameters
initial_lr = 0.001
epochs = 20
batch_size = 64

# Data augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest")

# Only rescaling for validation
val_datagen = ImageDataGenerator(rescale=1./255)

# Flow training images in batches using train_datagen generator
train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical')

# Flow validation images in batches using val_datagen generator
validation_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical')

# Build the model
model_base = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))
model_head = model_base.output
model_head = MaxPooling2D(pool_size=(5, 5))(model_head)
model_head = Flatten(name="flatten")(model_head)
model_head = Dense(128, activation="relu")(model_head)
model_head = Dropout(0.5)(model_head)
model_head = Dense(2, activation="softmax")(model_head)
model_final = Model(inputs=model_base.input, outputs=model_head)

# Freeze the base model layers
for layer in model_base.layers:
    layer.trainable = False

# Compile the model
optim = Adam(learning_rate=initial_lr)
model_final.compile(loss="categorical_crossentropy", optimizer=optim, metrics=["accuracy"])

# Calculate steps per epoch
steps_per_epoch = train_generator.samples // batch_size
validation_steps = validation_generator.samples // batch_size

# Train the model
model_train = model_final.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_steps)

# Predict on the validation data
validation_generator.reset()
num_validation_samples = validation_generator.samples
num_validation_steps = np.ceil(num_validation_samples / batch_size)

predictions = model_final.predict(validation_generator, steps=num_validation_steps)
predict_index = np.argmax(predictions, axis=1)

# Get the true labels
true_labels = validation_generator.classes[:len(predict_index)]

# Get the class labels
class_labels = list(validation_generator.class_indices.keys())

# Print classification report
print(classification_report(true_labels, predict_index, target_names=class_labels))
