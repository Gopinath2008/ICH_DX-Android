import os
import numpy as np
import pydicom
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models, Sequential
import matplotlib.pyplot as plt


def hu_normalization(image, slope, intercept):
    return image * slope + intercept

def window_image(image, window_center, window_width):
    min_hu = window_center - (window_width // 2)
    max_hu = window_center + (window_width // 2)

    image = np.clip(image, min_hu, max_hu)
    image = (image - min_hu) / (max_hu - min_hu)  # Normalize to [0,1]
    image = (image * 255).astype(np.uint8)  # Scale to [0,255]

    return image

def apply_sharpening(image):
    sharpening_kernel = np.array([[-1, -1, -1],
                                  [-1,  9, -1],
                                  [-1, -1, -1]])
    return cv2.filter2D(image, -1, sharpening_kernel)

def preprocess_dicom(file_path):
    try:
        dicom = pydicom.dcmread(file_path)
        image = dicom.pixel_array.astype(np.float32)

        # Apply HU normalization
        hu_image = hu_normalization(image, dicom.RescaleSlope, dicom.RescaleIntercept)

        # Apply windowing for Brain, Subdural, and Bone
        brain_window = window_image(hu_image, 40, 80)
        subdural_window = window_image(hu_image, 80, 200)
        bone_window = window_image(hu_image, 600, 2800)

        # Apply sharpening
        sharpened_brain = apply_sharpening(brain_window)
        sharpened_subdural = apply_sharpening(subdural_window)
        sharpened_bone = apply_sharpening(bone_window)

        # Merge the three windowed images into a single 3-channel image
        three_channel_image = cv2.merge([sharpened_brain, sharpened_subdural, sharpened_bone])

        # Resize to the desired image size
        three_channel_image = cv2.resize(three_channel_image, (256, 256))

        return three_channel_image
    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        return None  

class DICOMDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, dataset_path, batch_size=32, image_size=(256, 256), num_classes=5, sequence_length=1, shuffle=True):
        super().__init__()  
        
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_classes = num_classes
        self.sequence_length = sequence_length
        self.shuffle = shuffle

        self.class_names = sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])
        self.file_list = self._load_file_list()

        print(f"✅ Total files loaded: {len(self.file_list)}")

        if len(self.file_list) == 0:
            raise ValueError("❌ No DICOM files found. Check dataset path!")

        if self.shuffle:
            np.random.shuffle(self.file_list)

    def _load_file_list(self):
        file_list = []
        for class_name in self.class_names:
            class_path = os.path.join(self.dataset_path, class_name)
            dicom_files = sorted([os.path.join(class_path, f) for f in os.listdir(class_path) if f.endswith('.dcm')])

            if len(dicom_files) < self.sequence_length:
                print(f"⚠️ Skipping {class_name}: Not enough files ({len(dicom_files)})")
                continue  # Skip classes with insufficient data

            for i in range(len(dicom_files) - self.sequence_length + 1):
                file_list.append((dicom_files[i:i + self.sequence_length], class_name))

        return file_list

    def __len__(self):
        return max(1, len(self.file_list) // self.batch_size)

    def __getitem__(self, index):
        batch_files = self.file_list[index * self.batch_size:(index + 1) * self.batch_size]
        batch_images, batch_labels = [], []

        for file_paths, class_name in batch_files:
            sequence_images = []

            for file_path in file_paths:
                image = preprocess_dicom(file_path)
                if image is None:
                    continue 
                sequence_images.append(image)

            if len(sequence_images) != self.sequence_length:
                continue 

            sequence_images = np.stack(sequence_images, axis=0)

            if self.sequence_length == 1:
                sequence_images = sequence_images[0]  

            batch_images.append(sequence_images)

            label = self.class_names.index(class_name)
            batch_labels.append(tf.keras.utils.to_categorical(label, self.num_classes))

        if not batch_images or not batch_labels:
            print("⚠️ Empty batch encountered! Skipping...")
            return self.__getitem__((index + 1) % self.__len__())  

        batch_images = np.array(batch_images, dtype=np.float32)
        batch_labels = np.array(batch_labels, dtype=np.float32)

        return batch_images, batch_labels
      
    class DICOMDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, dataset_path, batch_size=32, image_size=(256, 256), num_classes=5, sequence_length=1, shuffle=True):
        super().__init__()  # Fixes Keras multiprocessing issue
        
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_classes = num_classes
        self.sequence_length = sequence_length
        self.shuffle = shuffle

        self.class_names = sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])
        self.file_list = self._load_file_list()

        print(f"✅ Total files loaded: {len(self.file_list)}")

        if len(self.file_list) == 0:
            raise ValueError("❌ No DICOM files found. Check dataset path!")

        if self.shuffle:
            np.random.shuffle(self.file_list)

    def _load_file_list(self):
        file_list = []
        for class_name in self.class_names:
            class_path = os.path.join(self.dataset_path, class_name)
            dicom_files = sorted([os.path.join(class_path, f) for f in os.listdir(class_path) if f.endswith('.dcm')])

            if len(dicom_files) < self.sequence_length:
                print(f"⚠️ Skipping {class_name}: Not enough files ({len(dicom_files)})")
                continue  # Skip classes with insufficient data

            for i in range(len(dicom_files) - self.sequence_length + 1):
                file_list.append((dicom_files[i:i + self.sequence_length], class_name))

        return file_list

    def __len__(self):
        return max(1, len(self.file_list) // self.batch_size)

    def __getitem__(self, index):
        """Generates a batch of data."""
        batch_files = self.file_list[index * self.batch_size:(index + 1) * self.batch_size]
        batch_images, batch_labels = [], []

        for file_paths, class_name in batch_files:
            sequence_images = []

            for file_path in file_paths:
                image = preprocess_dicom(file_path)
                if image is None:
                    continue  # Skip invalid images
                sequence_images.append(image)

            if len(sequence_imag
                   
                   es) != self.sequence_length:
                continue  # Ensure sequence length is met

            sequence_images = np.stack(sequence_images, axis=0)

            if self.sequence_length == 1:
                sequence_images = sequence_images[0]  # Shape: (256, 256, 3)

            batch_images.append(sequence_images)

            label = self.class_names.index(class_name)
            batch_labels.append(tf.keras.utils.to_categorical(label, self.num_classes))

        if not batch_images or not batch_labels:
            print("⚠️ Empty batch encountered! Skipping...")
            return self.__getitem__((index + 1) % self.__len__())  # Retry with next batch

        batch_images = np.array(batch_images, dtype=np.float32)
        batch_labels = np.array(batch_labels, dtype=np.float32)

        return batch_images, batch_labels

  import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
)
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(256, 256, 3)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(5, activation='softmax')  # Assuming 5 classes
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()



dataset_path = "C:/Users/Gopinath M/Music/Intracranial Hemorrhage Classification- DICOM"

train_generator = DICOMDataGenerator(dataset_path, batch_size=8, sequence_length=1)

history = model.fit(
    train_generator,
    epochs=50,
    steps_per_epoch=len(train_generator),
    verbose=1
)



import os


save_directory = r"C:\Users\Gopinath M\Music\Models"


os.makedirs(save_directory, exist_ok=True)


model_path = os.path.join(save_directory, "ICH-combined 2.h5")
model.save(model_path)

print(f"✅ Model saved successfully at: {model_path}")
