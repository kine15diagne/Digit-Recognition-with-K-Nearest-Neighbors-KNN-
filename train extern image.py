import matplotlib.pyplot as plt
import random
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from PIL import Image

# Load the digits dataset
data = datasets.load_digits()

# Define variables
X = data.images
y = data.target

# Flatten the input data since KNN expects 1D feature vectors
print(data.images.shape)  # Useful to understand the shape of the data
n_samples = len(X)
X = X.reshape((n_samples, -1))  # Reshape the 2D images into 1D vectors
print(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Initialize the KNN model
model = KNeighborsClassifier(n_neighbors=3)

# Train the model
model.fit(X_train, y_train)

# Predict the labels for the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy is {accuracy * 100:.2f}%")

""" Let's make a prediction using an external handwritten digit image """

# Load a handwritten digit image and convert it to grayscale
image = Image.open(" Image .png").convert("L")

# Display the image using matplotlib
plt.imshow(image, cmap="gray")  # Display in grayscale
plt.title("Image loaded with PIL")
plt.axis("off")  # Hide axes
plt.show()

# Resize the image to match the 8x8 format used during training
image_resized = image.resize((8, 8))

# Convert the image to a NumPy array and normalize pixel values
# Tip: You can inspect pixel range with image_array.min() and image_array.max()
image_array = np.array(image_resized, dtype=np.float32) / 16.0

# Flatten the image into a 1D vector
image_flattened = image_array.flatten().reshape(1, -1)
print(f"Image shape after preprocessing: {image_flattened}")

# Predict the digit using the trained model
predicted_label = model.predict(image_flattened)[0]
print(f"The model predicts the digit is: {predicted_label}")
print("Script executed successfully")
