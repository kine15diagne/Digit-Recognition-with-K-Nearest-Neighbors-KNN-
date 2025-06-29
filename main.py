#_________________________________________________________________________________________________________
#                       Digit Recognition with K-Nearest Neighbors (KNN)
#__________________________________________________________________________________________________________
import numpy as np
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from PIL import Image
import matplotlib.pyplot as plt

# Load the digits dataset
data = datasets.load_digits()
X = data.images
y = data.target

# Flatten the images
n_samples = len(X)
X = X.reshape((n_samples, -1))

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Train the K-Nearest Neighbors model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# Evaluate the model
accuracy = accuracy_score(y_test, model.predict(X_test))
print(f"Model accuracy: {accuracy * 100:.2f}%")

"""Prediction of a digit using an internal image from the dataset"""
# Select a random image from the training set
random_index = np.random.randint(0, len(X_train))
image_reshaped = X_train[random_index].reshape(8, 8)
true_label = y_train[random_index]

# Display the image
plt.imshow(image_reshaped, cmap="gray")
plt.axis("off")
plt.show()

# Predict the digit
image = X_train[random_index].reshape(1, -1)
predicted_label = model.predict(image)[0]  # [0] to extract the value from the array
print(f"The model predicts the digit is: {predicted_label}")
print(f"The true label is: {true_label}")
