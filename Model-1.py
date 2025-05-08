import os
import numpy as np
import pydicom
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models


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
    dicom = pydicom.dcmread(file_path)
    image = dicom.pixel_array.astype(np.float32)
    image = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-8)
    brain_window = apply_window(image, center=40, width=80)
    subdural_window = apply_window(image, center=50, width=150)
    bone_window = apply_window(image, center=600, width=2000)
    stacked_image = np.stack([brain_window, subdural_window, bone_window], axis=-1).astype(np.float32)
    stacked_image = cv2.resize(stacked_image, (256, 256))
    return stacked_image



def preprocess_dicom(dicom_path):
    dcm = pydicom.dcmread(dicom_path)
    image = dcm.pixel_array.astype(np.float32)
    hu_image = hu_normalization(image, dcm.RescaleSlope, dcm.RescaleIntercept)
    brain_window = window_image(hu_image, 40, 80)
    subdural_window = window_image(hu_image, 80, 200)
    bone_window = window_image(hu_image, 600, 2800)
    sharpened_brain = apply_sharpening(brain_window)
    sharpened_subdural = apply_sharpening(subdural_window)
    sharpened_bone = apply_sharpening(bone_window)
    three_channel_image = cv2.merge([sharpened_brain, sharpened_subdural, sharpened_bone])
    return sharpened_brain, sharpened_subdural, sharpened_bone, three_channel_image


dataset_path = "C:/Users/Gopinath M/Music/Intracranial Hemorrhage Classification- DICOM"
num_classes_to_display = 5  
processed_classes = 0  

for class_name in os.listdir(dataset_path):
    class_folder = os.path.join(dataset_path, class_name)
    if not os.path.isdir(class_folder):
        continue
    print(f"Processing class: {class_name}")  
    for filename in os.listdir(class_folder):
        if filename.endswith(".dcm"):
            dicom_path = os.path.join(class_folder, filename)
            sharpened_brain, sharpened_subdural, sharpened_bone, sharpened_three_channel = preprocess_dicom(dicom_path)
            fig, axes = plt.subplots(1, 4, figsize=(12, 5))
            axes[0].imshow(sharpened_brain, cmap='gray')
            axes[0].set_title("Brain Window")
            axes[0].axis("off")
            axes[1].imshow(sharpened_subdural, cmap='gray')
            axes[1].set_title("Subdural Window")
            axes[1].axis("off")
            axes[2].imshow(sharpened_bone, cmap='gray')
            axes[2].set_title("Bone Window")
            axes[2].axis("off")
            axes[3].imshow(sharpened_three_channel)
            axes[3].set_title("3-Channel Image")
            axes[3].axis("off")
            plt.show()
            processed_classes += 1  # Increment processed class count
            break  
    if processed_classes == num_classes_to_display:
        break




import tensorflow as tf
from tensorflow.keras import layers, models

def build_optimized_cnn_bigru(input_shape=(5, 256, 256, 3), num_classes=5):
    inputs = layers.Input(shape=input_shape)

    cnn = layers.TimeDistributed(layers.Conv2D(16, (3, 3), activation='relu', padding='same'))(inputs)
    cnn = layers.TimeDistributed(layers.Conv2D(16, (3, 3), activation='relu', padding='same', dilation_rate=2))(cnn)
    cnn = layers.TimeDistributed(layers.MaxPooling2D((2, 2)))(cnn)
    cnn = layers.TimeDistributed(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))(cnn)
    cnn = layers.TimeDistributed(layers.Conv2D(32, (3, 3), activation='relu', padding='same', dilation_rate=2))(cnn)
    cnn = layers.TimeDistributed(layers.MaxPooling2D((2, 2)))(cnn)
    fpn = layers.TimeDistributed(layers.Conv2D(64, (1, 1), activation='relu', padding='same'))(cnn)
    se = layers.GlobalAveragePooling3D()(fpn)
    se = layers.Dense(64, activation='relu')(se)
    se = layers.Dense(64, activation='sigmoid')(se)
    se = layers.Reshape((1, 1, 64))(se)
    cnn = layers.multiply([fpn, se])

    cnn = layers.TimeDistributed(layers.Flatten())(cnn)

    bigru = layers.Bidirectional(layers.GRU(64, return_sequences=True))(cnn)
    bigru = layers.BatchNormalization()(bigru)


    mhsa = layers.MultiHeadAttention(num_heads=4, key_dim=32)(bigru, bigru)
    mhsa = layers.LayerNormalization(epsilon=1e-6)(mhsa)  

    # 🔹 Fully Connected Layers
    x = layers.GlobalAveragePooling1D()(mhsa)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

l
model = build_optimized_cnn_bigru()
model.summary()





class DICOMDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, dataset_path, batch_size=32, image_size=(256, 256), num_classes=5, sequence_length=5, shuffle=True):
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

    def _preprocess_dicom(self, file_path):
        try:
            dicom = pydicom.dcmread(file_path)
            image = dicom.pixel_array.astype(np.float32)


            image = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-8)  
            image = (image * 255).astype(np.uint8)

            # Convert to 3-channel image
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

            # Resize image
            image = cv2.resize(image, self.image_size)

            return image
        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")
            return None  

    def __len__(self):
        return max(1, len(self.file_list) // self.batch_size)

    def __getitem__(self, index):
        batch_files = self.file_list[index * self.batch_size:(index + 1) * self.batch_size]
        batch_images, batch_labels = [], []

        for file_paths, class_name in batch_files:
            sequence_images = []

            for file_path in file_paths:
                image = self._preprocess_dicom(file_path)
                if image is None:
                    continue  
                sequence_images.append(image)

            if len(sequence_images) != self.sequence_length:
                continue  # Ensure sequence length is met

            batch_images.append(np.stack(sequence_images, axis=0))

            label = self.class_names.index(class_name)
            batch_labels.append(tf.keras.utils.to_categorical(label, self.num_classes))

        if not batch_images or not batch_labels:
            print("⚠️ Empty batch encountered! Skipping...")
            return self.__getitem__((index + 1) % self.__len__())  

        batch_images = np.array(batch_images, dtype=np.float32)
        batch_labels = np.array(batch_labels, dtype=np.float32)

        return batch_images, batch_labels



train_generator = DICOMDataGenerator(dataset_path, batch_size=32, sequence_length=5)


history = model.fit(
    train_generator,
    epochs=50,
    steps_per_epoch=len(train_generator),
    verbose=1,

)

from tensorflow.keras.models import load_model

model = load_model("hemorrhage_detection_model.h5")

print("✅ Model loaded successfully!")
model.save("hemorrhage_model")  
model = tf.keras.models.load_model("hemorrhage_model")  
