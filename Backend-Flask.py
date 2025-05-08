import os
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required
from flask_cors import CORS
from flask_mail import Mail, Message
import numpy as np
import cv2
import pydicom
import matplotlib
import matplotlib.pyplot as plt
import cv2
from scipy.ndimage import gaussian_filter
from tensorflow.keras.models import load_model
import smtplib
import random
from flask import Flask, send_from_directory

app = Flask(__name__)
CORS(app)

output_dir = "static/processed"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv("DATABASE_URI", "mysql+pymysql://root:@localhost/ich")
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "your_secret_key")


print(f"Database URI: {app.config['SQLALCHEMY_DATABASE_URI']}")


db = SQLAlchemy(app)


app.config['MAIL_SERVER'] = 'smtp.gmail.com'  
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'gn4614534@gmail.com'
app.config['MAIL_PASSWORD'] = 'bssw qknh hgfz jlzj'

mail = Mail(app)  


UPLOAD_FOLDER = "uploads"
PROCESSED_FOLDER = "processed"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["PROCESSED_FOLDER"] = PROCESSED_FOLDER

login_manager = LoginManager()
login_manager.init_app(app)


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(512), nullable=False)
    otp = db.Column(db.String(6), nullable=True)
    category = db.Column(db.String(50))


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


model1 = load_model(os.path.join("models", r"C:\Users\Gopinath M\Music\Models\ICH-combined 1.h5"), compile=False)
model2 = load_model(os.path.join("models", r"C:\Users\Gopinath M\Music\Models\ICH-combined 2.h5"), compile=False)


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

def preprocess_dicom(dicom_path, target_size=(256, 256)):
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
    model1_input = np.stack([three_channel_image] * 5, axis=0)  
    model1_input = np.expand_dims(model1_input, axis=0)  

    model2_input = np.expand_dims(three_channel_image, axis=0)  

    print("Model1 input shape:", model1_input.shape)  
    print("Model2 input shape:", model2_input.shape)  

    processed_image_filename = f"processed_{random.randint(1000, 9999)}.png"
    processed_image_path = os.path.join("static", "processed", processed_image_filename)
    plt.imsave(processed_image_path, three_channel_image)

    processed_image_url = f"http://192.168.106.133:5000/processed/{processed_image_filename}"

    return model1_input, model2_input, processed_image_path 


@app.route("/")
def home():
    return "Welcome to the ICH Prediction App!"



@app.route("/signup", methods=["POST"])
def signup():
    data = request.get_json()
    print("Received data:", data)  

    email = data.get("email")
    category = data.get("category") 

    if not email or not category:
        return jsonify({"error": "Missing email or category"}), 400

   
    print(f"Email: {email}, Category: {category}")

    
    valid_categories = {
        "Radiologists & Neurologists": "Radiologists and Neurologists",
        "Emergency Physicians": "Emergency Physicians",
        "Patients & Caregivers": "Patients and Caregivers"
    }
    
    
    if category in valid_categories:
        category = valid_categories[category]  

    
    if category not in valid_categories.values():
        return jsonify({"error": "Invalid category"}), 400

   
    user = User.query.filter_by(email=email).first()
    if user:
        return jsonify({"error": "User already exists"}), 409

    otp = str(random.randint(100000, 999999))

    # Save to database
    new_user = User(email=email, otp=otp, category=category)
    db.session.add(new_user)
    db.session.commit()

    # Send OTP via email
    try:
        msg = Message("Your OTP Code", sender=app.config['MAIL_USERNAME'], recipients=[email])
        msg.body = f"Your OTP code is: {otp}"
        mail.send(msg)
    except Exception as e:
        print(f"Email sending failed: {e}")
        return jsonify({"error": "Failed to send OTP"}), 500

    return jsonify({"message": "OTP sent successfully!"}), 201



from werkzeug.security import generate_password_hash, check_password_hash


@app.route("/validate_otp", methods=["POST"])
def validate_otp():
    try:
        data = request.get_json()
        email = data.get("email", "").strip()
        otp = data.get("otp", "").strip()
        password = data.get("password", "").strip()
        confirm_password = data.get("confirm_password", "").strip()

        if not email or not otp or not password or not confirm_password:
            return jsonify({"error": "All fields are required"}), 400

        if password != confirm_password:
            return jsonify({"error": "Passwords don't match"}), 400

       
        user = User.query.filter_by(email=email, otp=otp).first()
        if not user:
            return jsonify({"error": "Invalid OTP/Email"}), 400

        
        hashed_password = generate_password_hash(password)  
        print(f"🔑 Hashed Password Before Storing in DB: {hashed_password}")

        user.password = hashed_password
        user.otp = None  
        db.session.commit()

        print("✅ Registration successful! Hashed password stored.")
        return jsonify({"message": "Registration successful"}), 200

    except Exception as e:
        print(f"⚠️ Registration error: {e}")
        return jsonify({"error": "Registration failed"}), 500
    



@app.route("/request_password_reset", methods=["POST"])
def request_password_reset():
    data = request.get_json()
    email = data.get("email", "").strip()

    if not email:
        return jsonify({"success": False, "error": "Email is required"}), 400

    user = User.query.filter_by(email=email).first()
    if not user:
        return jsonify({"success": False, "error": "User not found"}), 404

    otp = str(random.randint(100000, 999999))
    user.otp = otp
    db.session.commit()

    try:
        msg = Message("Password Reset OTP", sender=app.config['MAIL_USERNAME'], recipients=[email])
        msg.body = f"Use this OTP to reset your password: {otp}"
        mail.send(msg)
    except Exception as e:
        print(f"Error sending OTP: {e}")
        return jsonify({"success": False, "error": "Failed to send OTP"}), 500

    return jsonify({"success": True, "message": "OTP sent to your email"}), 200





@app.route("/reset_password", methods=["POST"])
def reset_password():
    data = request.get_json()
    otp = data.get("otp", "").strip()
    new_password = data.get("new_password", "").strip()
    confirm_password = data.get("confirm_password", "").strip()

    if not all([otp, new_password, confirm_password]):
        return jsonify({"success": False, "error": "All fields are required"}), 400

    if new_password != confirm_password:
        return jsonify({"success": False, "error": "Passwords do not match"}), 400

    user = User.query.filter_by(otp=otp).first()
    if not user:
        return jsonify({"success": False, "error": "Invalid OTP"}), 400

    hashed_password = generate_password_hash(new_password)
    user.password = hashed_password
    user.otp = None
    db.session.commit()

    return jsonify({"success": True, "message": "Password has been reset successfully"}), 200








@app.route("/login", methods=["POST"])
def login():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid JSON data"}), 400  

        email = data.get("email", "").strip()
        password = data.get("password", "").strip()

        if not email or not password:
            return jsonify({"error": "Email and Password are required"}), 400  

        user = User.query.filter_by(email=email).first()
        if not user:
            return jsonify({"error": "Invalid credentials"}), 401  

        if check_password_hash(user.password, password):
            return jsonify({"success": True, "message": "Login successful"}), 200  
        else:
            return jsonify({"error": "Invalid credentials"}), 401  

    except Exception as e:
        print(f"⚠️ Login error: {e}")
        return jsonify({"error": "Server error. Please try again later."}), 500  
    
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

    
from flask import send_from_directory

@app.route("/processed/<filename>")
def get_processed_image(filename):
    return send_from_directory("static/processed", filename)



@app.route("/predict", methods=["POST"])

def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    # 🔍 Debugging: Print file metadata
    print(f"Received file: {file.filename}")
    print(f"File content type: {file.content_type}")

    try:
        # Ensure "uploads" directory exists
        os.makedirs("uploads", exist_ok=True)

        dicom_path = os.path.join("uploads", file.filename)
        file.save(dicom_path)

        # 🔍 Debugging: Check if file is saved properly
        if not os.path.exists(dicom_path):
            return jsonify({"error": "File not saved correctly"}), 500

        # 🔍 Debugging: Check if file is readable
        print(f"File saved at: {dicom_path}")
        print(f"File size: {os.path.getsize(dicom_path)} bytes")

        # Process the DICOM file
        model1_input, model2_input, processed_image_path = preprocess_dicom(dicom_path)

        print("Input to model1 shape:", model1_input.shape)
        print("Input to model2 shape:", model2_input.shape)

        # Perform classification
        predictions1 = model1.predict(model1_input)
        predictions2 = model2.predict(model2_input)
        avg_prediction = (predictions1 + predictions2) 

        predicted_label = class_labels[np.argmax(avg_prediction)]
        confidence = np.max(avg_prediction)

        if predicted_label not in class_labels:
            return jsonify({
                "Predicted Hemorrhage Type": "Normal",
                "Description": "No hemorrhage detected. The scan appears to be normal.",
                "Severity Level": "None",
                "Medical Suggestions": "No further action required. Routine follow-up recommended."
            })

        severity, recommendation = classify_severity(confidence)

        predicted_description = hemorrhage_descriptions.get(predicted_label, "Description not available")
        
        

        return jsonify({
            "image_url": processed_image_path,
            "Predicted Hemorrhage Type": predicted_label,
            "Description": predicted_description,
            "Severity Level": severity,
            "Medical Suggestions": recommendation,
        })

    except Exception as e:
        print(f"Error: {str(e)}")  # Log error in backend
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    os.makedirs("uploads", exist_ok=True)
    app.run(host="0.0.0.0", port=5000, debug=True)

