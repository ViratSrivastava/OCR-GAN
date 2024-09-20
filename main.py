import os
import json
from xml.etree import ElementTree as ET

from tensorflow.keras.models import load_model
import cv2

def preprocess_image(image_path, input_shape):
  """Preprocesses the image for the model."""
  img = cv2.imread(image_path, cv2.IMREAD_COLOR)
  img = cv2.resize(img, (input_shape[0], input_shape[1]))
  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
  img = img.astype("float32") / 255.0
  img = img.reshape(img.shape[0], img.shape[1], 1)  # Reshape for single channel
  return img

def predict_text(model, image_path, input_shape):
  """Predicts text from an image using the model."""
  image = preprocess_image(image_path, input_shape)
  prediction = model.predict(image[np.newaxis])  # Add axis for single image
  predicted_index = np.argmax(prediction, axis=1)[0]
  predicted_text = chr(predicted_index + ord('a'))  # Assuming alphabetical output
  return predicted_text

def save_text_output(text, output_dir, filename, formats=["json", "xml", "txt"]):
  """Saves the predicted text in different formats."""
  for format in formats:
    if format == "json":
      data = {"text": text}
      with open(os.path.join(output_dir, f"{filename}.json"), "w") as f:
        json.dump(data, f)
    elif format == "xml":
      root = ET.Element("text")
      root.text = text
      tree = ET.ElementTree(root)
      tree.write(os.path.join(output_dir, f"{filename}.xml"))
    elif format == "txt":
      with open(os.path.join(output_dir, f"{filename}.txt"), "w") as f:
        f.write(text)
    else:
      print(f"Unsupported format: {format}")

if __name__ == "__main__":
  # Model path
  model_path = "DNN\R2\ocr_model.h5"
  # Output directory
  output_dir = "output"
  # Image path (replace with yours input image)
  image_path = "images/image.jpg"

  # Load the model
  model = load_model(model_path)
  # Get model input shape
  input_shape = model.layers[0].input_shape[1:]

  # Predict text from the image
  predicted_text = predict_text(model, image_path, input_shape)

  # Create output directory if it doesn't exist
  os.makedirs(output_dir, exist_ok=True)

  # Save text output in different formats
  save_text_output(predicted_text, output_dir, os.path.splitext(os.path.basename(image_path))[0], formats=["json", "xml", "txt"])

  print(f"Predicted text: {predicted_text}")
  print(f"Text saved in: {output_dir}")