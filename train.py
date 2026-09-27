

import argparse
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from collections import Counter

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, help="Path to dataset")
args = parser.parse_args()

TRAIN_DIR = os.path.join(args.data_path, "train")
TEST_DIR = os.path.join(args.data_path, "test")

IMG_SIZE = (224, 224)
BATCH_SIZE = 16

train_datagen = ImageDataGenerator(
    rescale=1./255, rotation_range=10, width_shift_range=0.1,
    height_shift_range=0.1, zoom_range=0.1, horizontal_flip=True,
    validation_split=0.2
)
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode="categorical", subset="training"
)
validation_generator = train_datagen.flow_from_directory(
    TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode="categorical", subset="validation"
)

base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224,224,3))
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.4),
    layers.Dense(3, activation="softmax")
])

model.compile(optimizer=tf.keras.optimizers.Adam(0.0001),
              loss="categorical_crossentropy", metrics=["accuracy"])

counter = Counter(train_generator.classes)
total = sum(counter.values())
num_classes = len(counter)
class_weights = {cls: total / (num_classes * count) for cls, count in counter.items()}

model.fit(train_generator, validation_data=validation_generator,
          epochs=15, class_weight=class_weights)

os.makedirs("outputs", exist_ok=True)
model.save("outputs/covid_pneumonia_classifier.keras")