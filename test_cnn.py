import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Constants
picture_size = 48
folder_path = r"C:\Users\mmm\Downloads\LT_Survey\Project\My_Project\sorted_data"  # Path to your data
batch_size = 64

# Data Generators with Augmentation
datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, width_shift_range=0.2, 
                             height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, 
                             horizontal_flip=True, validation_split=0.2)

train_set = datagen.flow_from_directory(folder_path, target_size=(picture_size, picture_size),
                                        batch_size=batch_size, class_mode='binary', subset='training')

val_set = datagen.flow_from_directory(folder_path, target_size=(picture_size, picture_size),
                                      batch_size=batch_size, class_mode='binary', subset='validation')

# Load the VGG16 model with pre-trained weights, excluding the top layers
vgg = VGG16(weights='imagenet', include_top=False, input_shape=(picture_size, picture_size, 3))

# Freeze all layers initially
for layer in vgg.layers:
    layer.trainable = False

# Add custom layers on top of VGG16 for classification
x = Flatten()(vgg.output)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(1, activation='sigmoid')(x)

# Create the model by adding custom layers to the VGG16 base
model = Model(inputs=vgg.input, outputs=output)

# Compile the model with a low learning rate for transfer learning
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Set up callbacks for saving the best model and early stopping
callbacks = [ModelCheckpoint('./model_vgg16_finetuned.keras', save_best_only=True, monitor='val_accuracy', mode='max'),
             EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)]

# Train the model with the base layers frozen
history = model.fit(train_set, validation_data=val_set, epochs=50, callbacks=callbacks)

# Unfreeze the top layers of VGG16 for fine-tuning
for layer in vgg.layers[-4:]:  # Unfreeze the last 4 layers
    layer.trainable = True

# Recompile the model with a low learning rate to fine-tune the unfrozen layers
model.compile(optimizer=Adam(learning_rate=0.00001), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model again with the top layers unfrozen (fine-tuning)
history_finetune = model.fit(train_set, validation_data=val_set, epochs=50, callbacks=callbacks)

# Evaluate the model on the test data
test_set = datagen.flow_from_directory(folder_path, target_size=(picture_size, picture_size),
                                       batch_size=batch_size, class_mode='binary', shuffle=False)

test_loss, test_acc = model.evaluate(test_set)
print(f"Test accuracy: {test_acc}")#81-82%
print(f"Test loss: {test_loss}")#42-43%