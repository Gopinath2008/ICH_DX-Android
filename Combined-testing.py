import numpy as np
import cv2
import tensorflow as tf
import pydicom
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from tensorflow.keras.models import load_model


model1_path = r"C:\Users\Gopinath M\Music\Models\ICH-combined 1.h5"
model2_path = r"C:\Users\Gopinath M\Music\Models\ICH-combined 2.h5"

model1 = load_model(model1_path, compile=False)
model2 = load_model(model2_path, compile=False)


class_labels = ["Epidural", "Intraparenchymal", "Intraventricular", "Subarachnoid", "Subdural"]


hemorrhage_descriptions = {
    "Epidural": "A collection of blood between the skull and dura mater. Often due to trauma and may require surgery.",
    "Intraparenchymal": "Bleeding within brain tissue, usually from hypertension or trauma. May cause swelling and need intensive care.",
    "Intraventricular": "Bleeding into the brain's ventricles, affecting cerebrospinal fluid circulation. Can lead to hydrocephalus.",
    "Subarachnoid": "Bleeding in the space between the brain and meninges. Often caused by an aneurysm rupture, requiring urgent intervention.",
    "Subdural": "Blood accumulation between the dura and arachnoid layer. Common in head trauma, requiring possible surgical drainage."
}


def classify_severity(prediction_score):
    if prediction_score < 0.3:
        return "Mild", "Observation and follow-up recommended. No immediate intervention required."
    elif 0.3 <= prediction_score < 0.7:
        return "Moderate", "Monitoring in a hospital setting is advised. CT scans may be needed for progression assessment."
    else:
        return "Severe", "Immediate medical attention required. Possible surgical intervention needed."


def hu_normalization(image, slope, intercept):
    return image * slope + intercept

def window_image(image, window_center, window_width):
    window_min = window_center - window_width / 2
    window_max = window_center + window_width / 2
    windowed_image = np.clip(image, window_min, window_max)
    windowed_image = (windowed_image - window_min) / (window_max - window_min)
    return (windowed_image * 255).astype(np.uint8)

def apply_sharpening(image):
    blurred = gaussian_filter(image, sigma=1)
    sharpened = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)
    return sharpened

def preprocess_dicom(dicom_path, target_size=(256, 256), num_slices=5):
    dcm = pydicom.dcmread(dicom_path)
    image = dcm.pixel_array.astype(np.float32)

   
    hu_image = hu_normalization(image, dcm.RescaleSlope, dcm.RescaleIntercept)

 
    brain_window = window_image(hu_image, 40, 80)
    subdural_window = window_image(hu_image, 80, 200)
    bone_window = window_image(hu_image, 600, 2800)

  
    sharpened_brain = apply_sharpening(brain_window)
    sharpened_subdural = apply_sharpening(subdural_window)
    sharpened_bone = apply_sharpening(bone_window)

 
    sharpened_brain = cv2.resize(sharpened_brain, target_size)
    sharpened_subdural = cv2.resize(sharpened_subdural, target_size)
    sharpened_bone = cv2.resize(sharpened_bone, target_size)


    three_channel_image = cv2.merge([sharpened_brain, sharpened_subdural, sharpened_bone])


    image_sequence = np.stack([three_channel_image] * num_slices, axis=0)
    image_sequence = np.expand_dims(image_sequence, axis=0)

    single_image = np.expand_dims(three_channel_image, axis=0)

    return image_sequence, single_image, three_channel_image

def predict_hemorrhage(image_path):
    processed_sequence, processed_image, three_channel_image = preprocess_dicom(image_path)

    if processed_sequence is None or processed_image is None:
        print("Error: Could not process the image.")
        return


    preds1 = model1.predict(processed_sequence)[0]
    preds2 = model2.predict(processed_image)[0]


    final_preds = (preds1 + preds2) 
    predicted_class = np.argmax(final_preds)
    hemorrhage_type = class_labels[predicted_class]
    prediction_score = final_preds[predicted_class]


    severity, suggestion = classify_severity(prediction_score)

    plt.figure(figsize=(6, 6))
    plt.imshow(three_channel_image, cmap='gray')
    plt.axis("off")  
    plt.show()


    print(f"\033[1mPredicted Hemorrhage Type:\033[0m {hemorrhage_type}")  
    print(f"\033[1mDescription:\033[0m {hemorrhage_descriptions[hemorrhage_type]}")
    print(f"\033[1mSeverity Level:\033[0m {severity}")
    print(f"\033[1mMedical Suggestion:\033[0m {suggestion}")


image_path = r"C:\Users\Gopinath M\Music\Intracranial Hemorrhage Classification- DICOM\Subdural\ID_00c3c70f5.dcm"
predict_hemorrhage(image_path)
