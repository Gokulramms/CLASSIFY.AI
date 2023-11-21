import cv2
import numpy as np
from tensorflow.keras.models import load_model

def preprocess_image(img_path):
    img = cv2.imread(img_path)

    if img is None or img.size == 0:
        raise Exception(f"Error loading image from {img_path}")

    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype("float") / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def predict_image(img_array, model):
    prediction = model.predict(img_array)
    return "AI Generated" if prediction[0][0] >= 0.5 else "Real"

if __name__ == "__main__":
    # Load the pre-trained model
    model = load_model(r'c:\Users\Gokul Ramm\Downloads\ai_image_classifier (1).h5')  # Replace with the path to your model

    # Get user input for the image path
    image_path = r'c:\Users\Gokul Ramm\Pictures\Saved Pictures\IMG_20230421_205326.jpg'

    try:
        # Preprocess the image
        img_array = preprocess_image(image_path)

        # Make a prediction
        prediction_result = predict_image(img_array, model)

        # Display the result
        print(f"The image is likely {prediction_result}.")
    except Exception as e:
        print(f"Error processing the image: {e}")
