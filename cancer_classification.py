"""
Breast Cancer Classification with a simple Neural Network

---------------------------------------------------------
Runs in VS Code (plain Python file) - no Google Colab needed.

Run from the terminal:   python cancer_classification.py
Or in VS Code:           click "Run Python File" (the triangle at top-right).

What you get:
  1. All the results printed in the terminal (data info, training, accuracy,
     predictions, final "Malignant / Benign" result).
  2. A graph WINDOW that pops up with 3 charts:
        - Model accuracy
        - Model loss
        - Prediction result for the new patient (Malignant vs Benign)
  3. The graphs are also saved as PNG files next to this script.
"""

# %% 1. Import the libraries
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # hide TensorFlow's noisy info messages

import numpy as np
import pandas as pd
import matplotlib

# Use a real window (Tk) if available; otherwise matplotlib picks a default one
try:
    matplotlib.use("TkAgg")
except Exception:
    pass
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow import keras

tf.random.set_seed(3)
np.random.seed(3)

# Save graphs next to this script, wherever you run it from
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# %% 2. Load the data (built into scikit-learn - no file or download needed)
breast_cancer_dataset = load_breast_cancer()

data_frame = pd.DataFrame(
    breast_cancer_dataset.data, columns=breast_cancer_dataset.feature_names
)
data_frame["label"] = breast_cancer_dataset.target  # 1 = Benign, 0 = Malignant

# %% 3. Explore the data
print("First 5 rows:")
print(data_frame.head(), "\n")

print("Last 5 rows:")
print(data_frame.tail(), "\n")

print("Rows, columns:", data_frame.shape, "\n")

print("Missing values per column (all should be 0):")
print(data_frame.isnull().sum().sum(), "total missing values\n")

print("Statistics:")
print(data_frame.describe(), "\n")

print("Class counts (1 = Benign, 0 = Malignant):")
print(data_frame["label"].value_counts(), "\n")

print("Average of each feature per class:")
print(data_frame.groupby("label").mean(), "\n")

# %% 4. Separate features (X) and target (Y)
X = data_frame.drop(columns="label")
Y = data_frame["label"]

# %% 5. Split into training and testing data (80% / 20%)
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=2
)
print("Shapes (all, train, test):", X.shape, X_train.shape, X_test.shape, "\n")

# %% 6. Standardize the data (mean 0, std 1) - helps the network learn
scaler = StandardScaler()
# .values -> plain numpy arrays, which avoids the "feature names" warning later
X_train_std = scaler.fit_transform(X_train.values)
X_test_std = scaler.transform(X_test.values)

# %% 7. Build the neural network
model = keras.Sequential(
    [
        keras.Input(shape=(30,)),                # 30 input features
        keras.layers.Dense(20, activation="relu"),  # hidden layer
        keras.layers.Dense(2, activation="softmax"),  # 2 outputs: Malignant / Benign
    ]
)
model.summary()

# %% 8. Compile the network
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# %% 9. Train the network
history = model.fit(X_train_std, Y_train, validation_split=0.1, epochs=10)

# %% 10. Save accuracy and loss graphs as PNG files (the window is shown at the end)
plt.figure()
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("Model accuracy")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend(["training data", "validation data"], loc="lower right")
plt.savefig(os.path.join(OUTPUT_DIR, "accuracy_plot.png"))
plt.close()

plt.figure()
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Model loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["training data", "validation data"], loc="upper right")
plt.savefig(os.path.join(OUTPUT_DIR, "loss_plot.png"))
plt.close()

print("\nSaved accuracy_plot.png and loss_plot.png")

# %% 11. Accuracy on the test data
loss, accuracy = model.evaluate(X_test_std, Y_test)
print(f"\nTest accuracy: {accuracy:.4f}  ({accuracy * 100:.2f}%)")

# %% 12. Predictions on the test data
Y_pred = model.predict(X_test_std)  # probabilities for [Malignant, Benign]
Y_pred_labels = [int(np.argmax(p)) for p in Y_pred]
print("First 10 predicted labels:", Y_pred_labels[:10])
print("First 10 actual labels:   ", list(Y_test.values[:10]))

# %% 13. Predictive system: classify one new patient
input_data = (
    11.76, 21.6, 74.72, 427.9, 0.08637, 0.04966, 0.01657, 0.01115, 0.1495,
    0.05888, 0.4062, 1.21, 2.635, 28.47, 0.005857, 0.009758, 0.01168,
    0.007445, 0.02406, 0.001769, 12.98, 25.72, 82.98, 516.5, 0.1085,
    0.08615, 0.05523, 0.03715, 0.2433, 0.06563,
)

input_data_reshaped = np.asarray(input_data).reshape(1, -1)  # one data point
input_data_std = scaler.transform(input_data_reshaped)       # same scaling as training

prediction = model.predict(input_data_std)
prediction_label = int(np.argmax(prediction))
print("\nPrediction probabilities [Malignant, Benign]:", prediction)

if prediction_label == 0:
    print("The tumor is Malignant")
else:
    print("The tumor is Benign")

# %% 14. Show the graph window (accuracy, loss and the prediction result)
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Breast Cancer Prediction - Neural Network Results", fontsize=15)

# (a) accuracy
axes[0].plot(history.history["accuracy"], marker="o")
axes[0].plot(history.history["val_accuracy"], marker="o")
axes[0].set_title("Model accuracy")
axes[0].set_xlabel("epoch")
axes[0].set_ylabel("accuracy")
axes[0].legend(["training data", "validation data"], loc="lower right")
axes[0].grid(alpha=0.3)

# (b) loss
axes[1].plot(history.history["loss"], marker="o")
axes[1].plot(history.history["val_loss"], marker="o")
axes[1].set_title("Model loss")
axes[1].set_xlabel("epoch")
axes[1].set_ylabel("loss")
axes[1].legend(["training data", "validation data"], loc="upper right")
axes[1].grid(alpha=0.3)

# (c) prediction for the new patient
probs = prediction[0] * 100
bars = axes[2].bar(["Malignant", "Benign"], probs, color=["#d62728", "#2ca02c"])
axes[2].set_ylim(0, 110)
axes[2].set_ylabel("probability (%)")
result = "MALIGNANT" if prediction_label == 0 else "BENIGN"
axes[2].set_title(f"Prediction for new patient: {result}")
for bar, p in zip(bars, probs):
    axes[2].text(bar.get_x() + bar.get_width() / 2, p + 2, f"{p:.1f}%", ha="center")

fig.text(0.5, 0.01, f"Test accuracy: {accuracy * 100:.2f}%", ha="center", fontsize=12)
plt.tight_layout(rect=[0, 0.04, 1, 0.95])
fig.savefig(os.path.join(OUTPUT_DIR, "prediction_results.png"))

print("\nSaved prediction_results.png - close the graph window to finish.")
plt.show()  # opens the window
