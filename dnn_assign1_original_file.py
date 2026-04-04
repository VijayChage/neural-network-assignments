import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
DEBUG = True


def check_nan_inf(name, arr):
    if np.isnan(arr).any() or np.isinf(arr).any():
        print(f"WARNING: {name} contains NaN or Inf values!")

# ---------------------------------------------------------
# CUSTOM METRIC FUNCTIONS (From Scratch - Learning Focus)
# ---------------------------------------------------------


def mean_squared_error(y_true, y_pred):
    """Calculate MSE from scratch."""
    return np.mean((y_true - y_pred) ** 2)


def mean_absolute_error(y_true, y_pred):
    """Calculate MAE from scratch."""
    return np.mean(np.abs(y_true - y_pred))


def r2_score(y_true, y_pred):
    """Calculate R² score from scratch."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)


def confusion_matrix(y_true, y_pred):
    """Calculate confusion matrix from scratch."""
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    unique_labels = np.unique(np.concatenate([y_true, y_pred]))
    matrix = np.zeros((len(unique_labels), len(unique_labels)), dtype=int)

    for i, true_label in enumerate(unique_labels):
        for j, pred_label in enumerate(unique_labels):
            matrix[i, j] = np.sum((y_true == true_label)
                                  & (y_pred == pred_label))

    return matrix


def accuracy_score(y_true, y_pred):
    """Calculate accuracy from scratch."""
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    return np.mean(y_true == y_pred)


def precision_score(y_true, y_pred, average='macro', zero_division=0):
    """Calculate precision from scratch."""
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    if average == 'macro':
        unique_labels = np.unique(np.concatenate([y_true, y_pred]))
        precisions = []
        for label in unique_labels:
            tp = np.sum((y_true == label) & (y_pred == label))
            fp = np.sum((y_true != label) & (y_pred == label))
            if tp + fp == 0:
                precisions.append(zero_division)
            else:
                precisions.append(tp / (tp + fp))
        return np.mean(precisions)
    else:
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        return tp / (tp + fp) if (tp + fp) > 0 else zero_division


def recall_score(y_true, y_pred, average='macro', zero_division=0):
    """Calculate recall from scratch."""
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    if average == 'macro':
        unique_labels = np.unique(np.concatenate([y_true, y_pred]))
        recalls = []
        for label in unique_labels:
            tp = np.sum((y_true == label) & (y_pred == label))
            fn = np.sum((y_true == label) & (y_pred != label))
            if tp + fn == 0:
                recalls.append(zero_division)
            else:
                recalls.append(tp / (tp + fn))
        return np.mean(recalls)
    else:
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        return tp / (tp + fn) if (tp + fn) > 0 else zero_division


def f1_score(y_true, y_pred, average='macro', zero_division=0):
    """Calculate F1 score from scratch."""
    prec = precision_score(y_true, y_pred, average=average,
                           zero_division=zero_division)
    rec = recall_score(y_true, y_pred, average=average,
                       zero_division=zero_division)
    if prec + rec == 0:
        return zero_division
    return 2 * (prec * rec) / (prec + rec)

# ----------------------------------------------------
# Display details about Dataset chosen for the models
# ----------------------------------------------------


def get_assignment_results(dataset_name, n_samples, n_features, problem_type,
                           primary_metric, model1_name, model1_metrics,
                           model1_time, model1_arch, model2_name,
                           model2_metrics, model2_time, model2_arch):
    """
    Function to generate assignment results for any ML task.

    Parameters:
    -----------
    dataset_name : str
        Name of the dataset
    n_samples : int
        Number of samples
    n_features : int
        Number of features
    problem_type : str
        Type of problem (e.g., "Regression", "Binary Classification", "Multi-class Classification")
    primary_metric : str
        Primary evaluation metric
    model1_name : str
        Name of baseline model
    model1_metrics : dict
        Metrics dictionary for model 1
    model1_time : float
        Training time for model 1
    model1_arch : str
        Architecture description or layers for model 1
    model2_name : str
        Name of advanced model
    model2_metrics : dict
        Metrics dictionary for model 2
    model2_time : float
        Training time for model 2
    model2_arch : str
        Architecture description or layers for model 2

    Returns:
    --------
    tuple : (dataset_info, metrics_table, summary_table)
    """

    # Dataset Information Table
    dataset_info = pd.DataFrame({
        "Property": ["Dataset Name", "Samples", "Features", "Problem Type", "Primary Metric"],
        "Value": [dataset_name, n_samples, n_features, problem_type, primary_metric]
    })
    dataset_info.index = np.arange(1, len(dataset_info) + 1)

    # Metrics Comparison Table
    metric_names = list(model1_metrics.keys())
    metrics_table = pd.DataFrame({
        "Metric": metric_names,
        model1_name: [model1_metrics[m] for m in metric_names],
        model2_name: [model2_metrics[m] for m in metric_names]
    })
    metrics_table.iloc[:, 1:] = metrics_table.iloc[:, 1:].round(4)
    metrics_table.index = np.arange(1, len(metrics_table) + 1)

    # Model Summary Table
    summary_table = pd.DataFrame({
        "Model": [model1_name, model2_name],
        "Training Time (seconds)": [round(model1_time, 4), round(model2_time, 4)],
        "Architecture": [str(model1_arch), str(model2_arch)]
    })
    summary_table.index = np.arange(1, len(summary_table) + 1)

    return dataset_info, metrics_table, summary_table


# ---------------------------
# Common utility function
# ---------------------------
def plot_metrics_comparison(model1_metrics, model2_metrics, model1_name, model2_name):
    """Common function to plot metrics comparison for any two models."""
    labels = list(model1_metrics.keys())
    x = np.arange(len(labels))

    plt.figure(figsize=(10, 6))
    plt.bar(x - 0.2, list(model1_metrics.values()), 0.4, label=model1_name)
    plt.bar(x + 0.2, list(model2_metrics.values()), 0.4, label=model2_name)
    plt.xticks(x, labels)
    plt.ylabel("Score")
    plt.title(f"Model Comparison: {model1_name} vs {model2_name}")
    plt.legend()
    plt.grid(axis='y')
    plt.show()


def plot_loss_curves(model1_losses, model2_losses, model1_name, model2_name):
    """Common function to plot training loss curves for any two models."""
    plt.figure(figsize=(10, 6))
    plt.plot(model1_losses, label=model1_name)
    plt.plot(model2_losses, label=model2_name)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss Comparison")
    plt.legend()
    plt.grid()
    plt.show()


# ---------------------------------------------
# Task 1: Linear Regression - Housing Prices
# ---------------------------------------------

print("\n" + "="*50)
print("TASK 1: LINEAR REGRESSION - HOUSING PRICES")
print("="*50)

# 1.1 Choose Dataset
data = fetch_openml(name="house_prices", as_frame=True)

# 1.2 Preprocessing the data
X = data.data.select_dtypes(include=[np.number])
y = data.target.astype(float).values.reshape(-1, 1)

X = X.fillna(X.mean())

X_train, X_test, y_train, y_test = train_test_split(
    X.values, y, test_size=0.2, random_state=42
)

x_scaler = StandardScaler()
X_train = x_scaler.fit_transform(X_train)
X_test = x_scaler.transform(X_test)

y_scaler = StandardScaler()
y_train = y_scaler.fit_transform(y_train)
y_test = y_scaler.transform(y_test)

check_nan_inf("X_train", X_train)
check_nan_inf("X_test", X_test)
check_nan_inf("y_train", y_train)
check_nan_inf("y_test", y_test)

print("\n==============================")
print("Dataset Description")
print("==============================")
print(f"Samples: {X.shape[0]}")
print(f"Features: {X.shape[1]}")
print("Linear Regression")
print("Primary Metric: RMSE (penalizes large errors in house pricing)")
print("Secondary Metric: MAE, MSE & R2 Score (Others considered as secondary in a sequece mentioned)")

print("\nSplit Data Training:Test to 80:20 percentage")
print(f"\nTraining samples: {X_train.shape[0]}")
print(f"Test samples: {X_test.shape[0]}")


# 1.3 Apply Linear Regression logic
class LinearRegressionScratch:

    def __init__(self, lr=0.01, epochs=300):
        self.lr = lr
        self.epochs = epochs
        self.loss_history = []

    def initialize_weights(self, d):
        self.W = np.random.randn(d, 1) * 0.01
        self.b = 0

    def forward(self, X):
        return X @ self.W + self.b

    def compute_loss(self, y, y_pred):
        return np.mean((y - y_pred) ** 2)

    def compute_gradients(self, X, y, y_pred):
        n = X.shape[0]
        dW = (-2/n) * X.T @ (y - y_pred)
        db = (-2/n) * np.sum(y - y_pred)
        return dW, db

    def fit(self, X, y):
        n, d = X.shape
        self.initialize_weights(d)

        start_time = time.time()
        debug_rows = []

        for epoch in range(self.epochs):
            y_pred = self.forward(X)
            loss = self.compute_loss(y, y_pred)
            self.loss_history.append(loss)

            dW, db = self.compute_gradients(X, y, y_pred)

            dW = np.clip(dW, -1, 1)
            db = np.clip(db, -1, 1)

            if DEBUG and epoch % 50 == 0:
                debug_rows.append({
                    "Epoch": epoch,
                    "Loss": round(loss, 4),
                    "dW_mean": round(np.mean(dW), 6),
                    "db": round(float(db), 6),
                    "b_value": float(self.b)
                })

            self.W -= self.lr * dW
            self.b -= self.lr * db

        self.training_time = time.time() - start_time

        if DEBUG:
            print("\n=== Linear Regression Debug ===")
            print(pd.DataFrame(debug_rows).to_string(index=False))

        return self

    def predict(self, X):
        return self.forward(X)

# 1.4 Apply Multi-Layer Perceptron logic for Linear Regression


class MLP:

    def __init__(self, layers, lr=0.005):
        self.layers = layers
        self.lr = lr
        self.loss_history = []
        self.initialize_parameters()

    def initialize_parameters(self):
        self.W, self.b = [], []
        for i in range(len(self.layers) - 1):
            self.W.append(np.random.randn(
                self.layers[i], self.layers[i+1]) * np.sqrt(2/self.layers[i]))
            self.b.append(np.zeros((1, self.layers[i+1])))

    def relu(self, z):
        return np.maximum(0, z)

    def relu_deriv(self, z):
        return (z > 0).astype(float)

    def forward_propagation(self, X):
        self.A = [X]
        self.Z = []

        for i in range(len(self.W) - 1):
            z = self.A[-1] @ self.W[i] + self.b[i]
            self.Z.append(z)
            self.A.append(self.relu(z))

        z = self.A[-1] @ self.W[-1] + self.b[-1]
        self.Z.append(z)
        self.A.append(z)

        return z

    def compute_loss(self, y, y_pred):
        return np.mean((y - y_pred) ** 2)

    def backward_propagation(self, y):
        m = y.shape[0]
        dZ = (self.A[-1] - y)

        self.dW, self.db = [], []

        for i in reversed(range(len(self.W))):
            dW = self.A[i].T @ dZ / m
            db = np.sum(dZ, axis=0, keepdims=True) / m

            self.dW.insert(0, dW)
            self.db.insert(0, db)

            if i != 0:
                dZ = (dZ @ self.W[i].T) * self.relu_deriv(self.Z[i-1])

    def fit(self, X, y, epochs=300):
        start_time = time.time()
        debug_rows = []

        for epoch in range(epochs):
            y_pred = self.forward_propagation(X)
            loss = self.compute_loss(y, y_pred)
            self.loss_history.append(loss)

            self.backward_propagation(y)

            for i in range(len(self.dW)):
                self.dW[i] = np.clip(self.dW[i], -1, 1)
                self.db[i] = np.clip(self.db[i], -1, 1)

            if DEBUG and epoch % 50 == 0:
                row = {"Epoch": epoch, "Loss": round(loss, 4)}
                for i in range(len(self.dW)):
                    row[f"dW_L{i}"] = round(np.mean(self.dW[i]), 6)
                    row[f"b_L{i}"] = round(np.mean(self.b[i]), 6)
                debug_rows.append(row)

            for i in range(len(self.W)):
                self.W[i] -= self.lr * self.dW[i]
                self.b[i] -= self.lr * self.db[i]

        self.training_time = time.time() - start_time

        if DEBUG:
            print("\n=== MLP Debug ===")
            print(pd.DataFrame(debug_rows).to_string(index=False))

    def predict(self, X):
        return self.forward_propagation(X)


# 1.5 Train the Model
lin = LinearRegressionScratch().fit(X_train, y_train)

mlp = MLP([X_train.shape[1], 32, 16, 1], lr=0.01)
mlp.fit(X_train, y_train)

# 1.6 Evaluate the model


def inverse(y):
    return y_scaler.inverse_transform(y)


y_test_actual = inverse(y_test)
lin_pred = inverse(lin.predict(X_test))
mlp_pred = inverse(mlp.predict(X_test))


def metrics(y_true, y_pred):
    return {
        "MSE": mean_squared_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2 Score": r2_score(y_true, y_pred)
    }


lin_metrics = metrics(y_test_actual, lin_pred)
mlp_metrics = metrics(y_test_actual, mlp_pred)

for key in ["MSE", "RMSE", "MAE", "R2 Score"]:
    if key not in lin_metrics or key not in mlp_metrics:
        raise ValueError(f"Missing metric: {key}")

# 1.6 Linear Regression Metrics
df = pd.DataFrame([lin_metrics, mlp_metrics], index=[
                  "Linear Regression", "MLP"])
df["Training Time (s)"] = [lin.training_time, mlp.training_time]

print("\n========== MODEL PERFORMANCE TABLE ==========\n")
print(df.round(4).to_string())

plt.plot(lin.loss_history, label="Linear")
plt.plot(mlp.loss_history, label="MLP")
plt.legend()
plt.title("Loss Curve")
plt.show()

# ---------------------------------------
# 1.7 Compare Linear Regression and MLP
# ---------------------------------------

metrics_list = ["MSE", "RMSE", "MAE", "R2 Score"]

linear_vals = [lin_metrics[m] for m in metrics_list]
mlp_vals = [mlp_metrics[m] for m in metrics_list]

for i, metric in enumerate(metrics_list):
    plt.figure()
    plt.bar(["Linear", "MLP"], [linear_vals[i], mlp_vals[i]])
    plt.title(f"{metric} Comparison")
    plt.grid(axis='y')
    plt.show()

# ----------------------
# 1.8 Compare RMSE and MAE
# ----------------------

labels = ["RMSE", "MAE"]
x = np.arange(len(labels))

plt.figure()
plt.bar(x - 0.2, [lin_metrics[l] for l in labels], 0.4, label="Linear")
plt.bar(x + 0.2, [mlp_metrics[l] for l in labels], 0.4, label="MLP")
plt.xticks(x, labels)
plt.title("RMSE & MAE Comparison")
plt.legend()
plt.grid(axis='y')
plt.show()

# Use unified function
data_info, metrics_table, summary_table = get_assignment_results(
    dataset_name="House Prices",
    n_samples=X.shape[0],
    n_features=X.shape[1],
    problem_type="Regression",
    primary_metric="RMSE",
    model1_name="Linear Regression",
    model1_metrics=lin_metrics,
    model1_time=lin.training_time,
    model1_arch=f"Single layer (d={X.shape[1]})",
    model2_name="MLP",
    model2_metrics=mlp_metrics,
    model2_time=mlp.training_time,
    model2_arch=mlp.layers
)

print("\n========== DATASET INFO ==========\n")
print(data_info.to_string(index=False))

print("\n========== BASELINE MODEL (Linear Regression) ==========\n")
print(metrics_table.to_string())

print("\n========== MODEL SUMMARY ==========\n")
print(summary_table.to_string())

print("\n========== FINAL ANALYSIS ==========\n")

rmse_diff = lin_metrics["RMSE"] - mlp_metrics["RMSE"]
mae_diff = lin_metrics["MAE"] - mlp_metrics["MAE"]
r2_diff = mlp_metrics["R2 Score"] - lin_metrics["R2 Score"]

better_model = "MLP" if rmse_diff > 0 else "Linear Regression"

analysis = f"""
{better_model} performs better on this regression task.

RMSE improves by {abs(rmse_diff):.2f} and MAE by {abs(mae_diff):.2f}, indicating more accurate predictions. 
R² increases by {r2_diff:.4f}, meaning better variance explanation.

Linear Regression is faster and simpler but assumes linear relationships. 
MLP captures non-linear patterns using hidden layers, leading to better performance.

The main challenge is tuning learning rate and stabilizing gradients.

Overall, MLP is preferred for accuracy, while Linear Regression is useful as a fast baseline.
"""

print(analysis.strip())


# -------------------------------------------------------------
# TASK 2: Logistic Regression - BREAST CANCER CLASSIFICATION
# -------------------------------------------------------------

print("\n" + "="*50)
print("TASK 2: BREAST CANCER CLASSIFICATION")
print("="*50)

# 2.1 Choose Dataset
data = load_breast_cancer()

# 2.2 Preprocessing the data
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

X = X.fillna(X.mean())

print("\nDataset Source: UCI Machine Learning Repository (via sklearn)")

dataset_name = "Breast Cancer Dataset"

print("Dataset Name:", dataset_name)
print("Description: This dataset contains medical diagnostic features computed from breast mass images.")
print("Goal: Predict whether a tumor is malignant (cancerous) or benign (non-cancerous).")

print("Number of Samples:", X.shape[0])
print("Number of Features:", X.shape[1])
print("Problem Type: Binary Classification")
print("Primary Metric: F1-score")

print("\nReason for choosing F1-score:")
print("It balances precision and recall, which is important in medical diagnosis.")

X_train, X_test, y_train, y_test = train_test_split(
    X.values, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

check_nan_inf("X_train", X_train)
check_nan_inf("X_test", X_test)
check_nan_inf("y_train", y_train)
check_nan_inf("y_test", y_test)

y_train = y_train.values.reshape(-1, 1)
y_test = y_test.values.reshape(-1, 1)

print("\n[DEBUG INFO]")
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("Any NaNs in X_train?", np.isnan(X_train).sum())

# 2.3 Apply Logistic Regression logic


class LogisticRegressionScratch:
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.loss_history = []

    def sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def compute_loss(self, y, y_hat):
        return -np.mean(y*np.log(y_hat + 1e-8) + (1-y)*np.log(1-y_hat + 1e-8))

    def fit(self, X, y):
        m, n = X.shape

        self.w = np.zeros((n, 1))
        self.b = 0

        start_time = time.time()

        for epoch in range(self.epochs):
            z = np.dot(X, self.w) + self.b
            y_hat = self.sigmoid(z)

            loss = self.compute_loss(y, y_hat)
            self.loss_history.append(loss)

            dw = (1/m) * np.dot(X.T, (y_hat - y))
            db = (1/m) * np.sum(y_hat - y)

            self.w -= self.lr * dw
            self.b -= self.lr * db

            if epoch % 200 == 0:
                print(f"Epoch {epoch} | Loss: {loss:.4f}")
        self.training_time = time.time() - start_time

    def predict(self, X):
        probs = self.sigmoid(np.dot(X, self.w) + self.b)
        return (probs > 0.5).astype(int)

# 2.4 Apply Multi-Layer Perceptron logic for Binary Classification


class MLPBinaryClassifier:
    def __init__(self, layers, lr=0.005, epochs=1500):
        self.layers = layers
        self.lr = lr
        self.epochs = epochs
        self.loss_history = []
        self.initialize_parameters()

    def initialize_parameters(self):
        self.params = {}
        for i in range(1, len(self.layers)):
            self.params["W"+str(i)] = np.random.randn(
                self.layers[i-1], self.layers[i]
            ) * np.sqrt(2 / self.layers[i-1])
            self.params["b"+str(i)] = np.zeros((1, self.layers[i]))

    def relu(self, Z):
        return np.maximum(0, Z)

    def relu_derivative(self, Z):
        return (Z > 0).astype(float)

    def sigmoid(self, Z):
        Z = np.clip(Z, -500, 500)
        return 1 / (1 + np.exp(-Z))

    def compute_loss(self, y, y_hat):
        return -np.mean(y*np.log(y_hat + 1e-8) + (1-y)*np.log(1-y_hat + 1e-8))

    def forward_propagation(self, X):
        self.cache = {"A0": X}

        for i in range(1, len(self.layers)):
            Z = np.dot(self.cache["A"+str(i-1)],
                       self.params["W"+str(i)]) + self.params["b"+str(i)]

            if i == len(self.layers)-1:
                A = self.sigmoid(Z)
            else:
                A = self.relu(Z)

            self.cache["Z"+str(i)] = Z
            self.cache["A"+str(i)] = A

        return A

    def backward_propagation(self, y):
        grads = {}
        m = y.shape[0]
        L = len(self.layers) - 1

        dZ = self.cache["A"+str(L)] - y

        for i in reversed(range(1, L+1)):
            grads["dW"+str(i)] = (1/m) * np.dot(self.cache["A"+str(i-1)].T, dZ)
            grads["db"+str(i)] = (1/m) * np.sum(dZ, axis=0, keepdims=True)

            if i > 1:
                dA = np.dot(dZ, self.params["W"+str(i)].T)
                dZ = dA * self.relu_derivative(self.cache["Z"+str(i-1)])

        return grads

    def update_parameters(self, grads):
        for i in range(1, len(self.layers)):
            self.params["W"+str(i)] -= self.lr * grads["dW"+str(i)]
            self.params["b"+str(i)] -= self.lr * grads["db"+str(i)]

    def fit(self, X, y):
        self.initialize_parameters()
        start_time = time.time()

        for epoch in range(self.epochs):
            y_hat = self.forward_propagation(X)
            loss = self.compute_loss(y, y_hat)
            self.loss_history.append(loss)

            grads = self.backward_propagation(y)
            self.update_parameters(grads)

            if epoch % 200 == 0:
                print(f"Epoch {epoch} | Loss: {loss:.4f}")

        self.training_time = time.time() - start_time

    def predict(self, X):
        probs = self.forward_propagation(X)
        return (probs > 0.5).astype(int)


print("\n=== Logistic Regression ===")
lr_model = LogisticRegressionScratch()
lr_model.fit(X_train, y_train)

print("\n=== MLP ===")
mlp2 = MLPBinaryClassifier(layers=[X_train.shape[1], 32, 16, 1])
mlp2.fit(X_train, y_train)


def compute_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = cm.ravel()

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "TP": TP, "TN": TN, "FP": FP, "FN": FN
    }


lr_pred = lr_model.predict(X_test)
mlp_pred = mlp2.predict(X_test)

lr_metrics = compute_metrics(y_test, lr_pred)
mlp_metrics = compute_metrics(y_test, mlp_pred)


def plot_confusion_matrix(cm, title):
    plt.imshow(cm)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha='center', va='center')

    plt.show()


plot_confusion_matrix(confusion_matrix(y_test, lr_pred),
                      "Logistic Regression Confusion Matrix")
plot_confusion_matrix(confusion_matrix(
    y_test, mlp_pred), "MLP Confusion Matrix")

# Use unified function for Task 2
dataset_info, metrics_table, summary_table = get_assignment_results(
    dataset_name="Breast Cancer Dataset",
    n_samples=X.shape[0],
    n_features=X.shape[1],
    problem_type="Binary Classification",
    primary_metric="F1-score",
    model1_name="Logistic Regression",
    model1_metrics={k: v for k, v in lr_metrics.items(
    ) if k in ["accuracy", "precision", "recall", "f1"]},
    model1_time=lr_model.training_time,
    model1_arch="Linear Model",
    model2_name="MLP",
    model2_metrics={k: v for k, v in mlp_metrics.items(
    ) if k in ["accuracy", "precision", "recall", "f1"]},
    model2_time=mlp2.training_time,
    model2_arch=mlp2.layers
)

print("\n========== DATASET INFORMATION ==========\n")
print(dataset_info.to_string(index=False))

print("\n========== MODEL PERFORMANCE COMPARISON ==========\n")
print(metrics_table.to_string())

print("\n========== MODEL DETAILS ==========\n")
print(summary_table.to_string())

plt.plot(lr_model.loss_history, label="Logistic Regression")
plt.plot(mlp2.loss_history, label="MLP")
plt.legend()
plt.title("Training Loss Comparison")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()

labels = ["accuracy", "precision", "recall", "f1"]

lr_vals = [lr_metrics[k] for k in labels]
mlp_vals = [mlp_metrics[k] for k in labels]

x = np.arange(len(labels))

plt.bar(x-0.2, lr_vals, 0.4, label="LR")
plt.bar(x+0.2, mlp_vals, 0.4, label="MLP")

plt.xticks(x, labels)
plt.legend()
plt.title("Model Comparison")
plt.show()

print("\n ANALYSIS\n")

f1_diff = mlp_metrics["f1"] - lr_metrics["f1"]
acc_diff = mlp_metrics["accuracy"] - lr_metrics["accuracy"]

print(f"• Logistic Regression F1-score: {lr_metrics['f1']:.4f}")
print(f"• MLP F1-score: {mlp_metrics['f1']:.4f}")

if f1_diff > 0:
    print(
        f"• MLP outperforms Logistic Regression by {f1_diff:.4f} in F1-score.")
elif f1_diff < 0:
    print(
        f"• Logistic Regression outperforms MLP by {-f1_diff:.4f} in F1-score.")
else:
    print("• Both models achieved identical F1-score.")

print(f"• Accuracy difference: {acc_diff:.4f}")

print(f"• Logistic Regression training time: {lr_model.training_time:.4f}s")
print(f"• MLP training time: {mlp2.training_time:.4f}s")

print("• Logistic Regression performs well due to near linear separability of dataset.")
print("• MLP captures non-linearity but does not significantly improve performance.")
print("• MLP is computationally more expensive due to multiple layers.")
print("• Key challenge: tuning learning rate and hidden layer sizes.")
print("• Insight: simpler models can outperform complex models on structured data.")


# ---------------------------------------------------------------------
# TASK 3: VEHICLE MULTI-CLASS CLASSIFICATION
# ---------------------------------------------------------------------

print("\n" + "="*50)
print("TASK 3: VEHICLE MULTI-CLASS CLASSIFICATION")
print("="*50)

print("\nLoading dataset...")
data = fetch_openml(name='vehicle', version=1, as_frame=True)

X = data.data
y = pd.factorize(data.target)[0]

dataset_name = "Vehicle Dataset (Multi-class)"

print("\nDataset:", dataset_name)
print("Samples:", X.shape[0])
print("Features:", X.shape[1])
print("Classes:", len(np.unique(y)))

print("\nFeature Types:")
print(X.dtypes.value_counts())

print("\nClass Distribution:")
print(pd.Series(y).value_counts())

print("\nPrimary Metric Justification:")
print("F1-score is used because class distribution is slightly imbalanced.")

X = X.fillna(X.mean())

X_train, X_test, y_train, y_test = train_test_split(
    X.values, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


def one_hot(y, k):
    o = np.zeros((len(y), k))
    o[np.arange(len(y)), y] = 1
    return o


num_classes = len(np.unique(y))
y_train_ohe = one_hot(y_train, num_classes)


class SoftmaxRegression:
    def __init__(self, lr=0.01, epochs=200):
        self.lr = lr
        self.epochs = epochs
        self.loss_history = []

    def softmax(self, z):
        z -= np.max(z, axis=1, keepdims=True)
        exp = np.exp(z)
        return exp / np.sum(exp, axis=1, keepdims=True)

    def fit(self, X, y):
        n, d = X.shape
        k = y.shape[1]

        self.W = np.zeros((d, k))
        self.b = np.zeros((1, k))

        for _ in range(self.epochs):
            logits = X @ self.W + self.b
            probs = self.softmax(logits)

            loss = -np.mean(np.sum(y * np.log(probs + 1e-8), axis=1))
            self.loss_history.append(loss)

            dZ = (probs - y) / n
            dW = X.T @ dZ
            db = np.sum(dZ, axis=0, keepdims=True)

            self.W -= self.lr * dW
            self.b -= self.lr * db

    def predict(self, X):
        probs = self.softmax(X @ self.W + self.b)
        return np.argmax(probs, axis=1)


class MLPMultiClass:
    def __init__(self, layers, lr=0.01):
        self.layers = layers
        self.lr = lr
        self.loss_history = []
        self.initialize_parameters()

    def initialize_parameters(self):
        self.W, self.b = [], []
        for i in range(len(self.layers) - 1):
            self.W.append(np.random.randn(
                self.layers[i], self.layers[i+1]) * np.sqrt(2/self.layers[i]))
            self.b.append(np.zeros((1, self.layers[i+1])))

    def relu(self, z): return np.maximum(0, z)
    def relu_deriv(self, z): return (z > 0).astype(float)

    def softmax(self, z):
        z -= np.max(z, axis=1, keepdims=True)
        exp = np.exp(z)
        return exp / np.sum(exp, axis=1, keepdims=True)

    def forward_propagation(self, X):
        self.A, self.Z = [X], []

        for i in range(len(self.W)-1):
            z = self.A[-1] @ self.W[i] + self.b[i]
            self.Z.append(z)
            self.A.append(self.relu(z))

        z = self.A[-1] @ self.W[-1] + self.b[-1]
        self.Z.append(z)
        self.A.append(self.softmax(z))
        return self.A[-1]

    def backward_propagation(self, y):
        m = y.shape[0]
        dZ = self.A[-1] - y
        self.dW, self.db = [], []

        for i in reversed(range(len(self.W))):
            dW = self.A[i].T @ dZ / m
            db = np.sum(dZ, axis=0, keepdims=True) / m

            self.dW.insert(0, dW)
            self.db.insert(0, db)

            if i != 0:
                dZ = (dZ @ self.W[i].T) * self.relu_deriv(self.Z[i-1])

    def fit(self, X, y, epochs=200):
        for _ in range(epochs):
            probs = self.forward_propagation(X)
            loss = -np.mean(np.sum(y * np.log(probs + 1e-8), axis=1))
            self.loss_history.append(loss)

            self.backward_propagation(y)

            for i in range(len(self.W)):
                self.W[i] -= self.lr * self.dW[i]
                self.b[i] -= self.lr * self.db[i]

    def predict(self, X):
        return np.argmax(self.forward_propagation(X), axis=1)


softmax_model = SoftmaxRegression()
mlp3 = MLPMultiClass([X_train.shape[1], 32, num_classes])

start = time.time()
softmax_model.fit(X_train, y_train_ohe)
softmax_time = time.time() - start

start = time.time()
mlp3.fit(X_train, y_train_ohe)
mlp_time = time.time() - start


def metrics_mc(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average='macro', zero_division=0),
        "Recall": recall_score(y_true, y_pred, average='macro', zero_division=0),
        "F1 Score": f1_score(y_true, y_pred, average='macro', zero_division=0)
    }


y_pred_softmax = softmax_model.predict(X_test)
y_pred_mlp3 = mlp3.predict(X_test)

softmax_metrics = metrics_mc(y_test, y_pred_softmax)
mlp_metrics3 = metrics_mc(y_test, y_pred_mlp3)

results_df = pd.DataFrame(
    [softmax_metrics, mlp_metrics3], index=["Softmax", "MLP"])
print("\n========== PERFORMANCE ==========\n")
print(results_df.round(4))

plt.figure()
plt.plot(softmax_model.loss_history, label="Softmax")
plt.plot(mlp3.loss_history, label="MLP")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss Comparison")
plt.legend()
plt.grid()
plt.show()

labels = list(softmax_metrics.keys())
x = np.arange(len(labels))

plt.figure()
plt.bar(x - 0.2, list(softmax_metrics.values()), 0.4, label="Softmax")
plt.bar(x + 0.2, list(mlp_metrics3.values()), 0.4, label="MLP")
plt.xticks(x, labels)
plt.ylabel("Score")
plt.title("Model Comparison")
plt.legend()
plt.grid(axis='y')
plt.show()

print("\nAnalysis:")

f1_diff = mlp_metrics3["F1 Score"] - softmax_metrics["F1 Score"]

print(f"MLP improves F1-score by {f1_diff:.4f}.")
print(f"MLP Training Time: {mlp_time:.2f}s vs Softmax: {softmax_time:.2f}s.")

if f1_diff > 0:
    print("MLP performs better due to non-linear learning capability.")
else:
    print("Softmax performs similarly due to linear separability.")

print("MLP is computationally more expensive than Softmax.")
print("Overall, trade-off exists between accuracy and computation.")

# Use unified function for Task 3
results_dict = get_assignment_results(
    dataset_name="Vehicle Dataset (Multi-class)",
    n_samples=X.shape[0],
    n_features=X.shape[1],
    problem_type="Multi-class Classification",
    primary_metric="F1-macro",
    model1_name="Softmax Regression",
    model1_metrics=softmax_metrics,
    model1_time=softmax_time,
    model1_arch="Linear classifier",
    model2_name="MLP",
    model2_metrics=mlp_metrics3,
    model2_time=mlp_time,
    model2_arch=mlp3.layers
)

print("\n" + "="*50)
print("ALL TASKS COMPLETED")
print("="*50)
