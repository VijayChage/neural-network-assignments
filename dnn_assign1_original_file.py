import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

np.random.seed(42)
DEBUG = True


def check_nan_inf(name, arr):
    if np.isnan(arr).any() or np.isinf(arr).any():
        print(f"WARNING: {name} contains NaN or Inf values!")


# -----------------------------
# Task 1: House Prices Regression
# -----------------------------
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
        dW = (-2 / n) * X.T @ (y - y_pred)
        db = (-2 / n) * np.sum(y - y_pred)
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
                    "b_value": float(self.b),
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


class MLPRegressor:
    def __init__(self, layers, lr=0.005):
        self.layers = layers
        self.lr = lr
        self.loss_history = []
        self.initialize_parameters()

    def initialize_parameters(self):
        self.W, self.b = [], []
        for i in range(len(self.layers) - 1):
            self.W.append(np.random.randn(
                self.layers[i], self.layers[i+1]) * np.sqrt(2 / self.layers[i]))
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
        dZ = self.A[-1] - y
        self.dW, self.db = [], []

        for i in reversed(range(len(self.W))):
            dW = self.A[i].T @ dZ / m
            db = np.sum(dZ, axis=0, keepdims=True) / m
            self.dW.insert(0, dW)
            self.db.insert(0, db)
            if i != 0:
                dZ = (dZ @ self.W[i].T) * self.relu_deriv(self.Z[i - 1])

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


def run_house_prices_regression():
    data = fetch_openml(name="house_prices", as_frame=True)
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

    print("\nSplit Data Training:Test to 80:20 percentage")
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")

    lin = LinearRegressionScratch().fit(X_train, y_train)
    mlp = MLPRegressor([X_train.shape[1], 32, 16, 1], lr=0.01)
    mlp.fit(X_train, y_train)

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
            "R2 Score": r2_score(y_true, y_pred),
        }

    lin_metrics = metrics(y_test_actual, lin_pred)
    mlp_metrics = metrics(y_test_actual, mlp_pred)

    for key in ["MSE", "RMSE", "MAE", "R2 Score"]:
        if key not in lin_metrics or key not in mlp_metrics:
            raise ValueError(f"Missing metric: {key}")

    df = pd.DataFrame([lin_metrics, mlp_metrics], index=[
                      "Linear Regression", "MLP"])
    df["Training Time (s)"] = [lin.training_time, mlp.training_time]

    print("\n========== MODEL PERFORMANCE TABLE ==========")
    print(df.round(4).to_string())

    plt.plot(lin.loss_history, label="Linear")
    plt.plot(mlp.loss_history, label="MLP")
    plt.legend()
    plt.title("Loss Curve")
    plt.show()

    metrics_list = ["MSE", "RMSE", "MAE", "R2 Score"]
    for metric in metrics_list:
        plt.figure()
        plt.bar(["Linear", "MLP"], [lin_metrics[metric], mlp_metrics[metric]])
        plt.title(f"{metric} Comparison")
        plt.grid(axis="y")
        plt.show()

    plt.figure()
    labels = ["RMSE", "MAE"]
    x = np.arange(len(labels))
    plt.bar(x - 0.2, [lin_metrics[l] for l in labels], 0.4, label="Linear")
    plt.bar(x + 0.2, [mlp_metrics[l] for l in labels], 0.4, label="MLP")
    plt.xticks(x, labels)
    plt.title("RMSE & MAE Comparison")
    plt.legend()
    plt.grid(axis="y")
    plt.show()

    rmse_diff = lin_metrics["RMSE"] - mlp_metrics["RMSE"]
    mae_diff = lin_metrics["MAE"] - mlp_metrics["MAE"]
    r2_diff = mlp_metrics["R2 Score"] - lin_metrics["R2 Score"]
    better_model = "MLP" if rmse_diff > 0 else "Linear Regression"

    print("\n========== FINAL ANALYSIS ==========")
    print(f"{better_model} performs better on this regression task.")
    print(
        f"RMSE improves by {abs(rmse_diff):.2f} and MAE by {abs(mae_diff):.2f}.")
    print(f"R2 increases by {r2_diff:.4f}.")
    print("Linear Regression is faster and simpler, while MLP captures non-linear patterns.")


# -----------------------------
# Task 2: Breast Cancer Classification
# -----------------------------
class LogisticRegressionScratch:
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.loss_history = []

    def sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def compute_loss(self, y, y_hat):
        return -np.mean(y * np.log(y_hat + 1e-8) + (1 - y) * np.log(1 - y_hat + 1e-8))

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
            dw = (1 / m) * np.dot(X.T, (y_hat - y))
            db = (1 / m) * np.sum(y_hat - y)
            self.w -= self.lr * dw
            self.b -= self.lr * db
            if epoch % 200 == 0:
                print(f"Epoch {epoch} | Loss: {loss:.4f}")

        self.training_time = time.time() - start_time

    def predict(self, X):
        probs = self.sigmoid(np.dot(X, self.w) + self.b)
        return (probs > 0.5).astype(int)


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
            self.params[f"W{i}"] = np.random.randn(
                self.layers[i-1], self.layers[i]) * np.sqrt(2 / self.layers[i-1])
            self.params[f"b{i}"] = np.zeros((1, self.layers[i]))

    def relu(self, Z):
        return np.maximum(0, Z)

    def relu_derivative(self, Z):
        return (Z > 0).astype(float)

    def sigmoid(self, Z):
        Z = np.clip(Z, -500, 500)
        return 1 / (1 + np.exp(-Z))

    def compute_loss(self, y, y_hat):
        return -np.mean(y * np.log(y_hat + 1e-8) + (1 - y) * np.log(1 - y_hat + 1e-8))

    def forward_propagation(self, X):
        self.cache = {"A0": X}
        for i in range(1, len(self.layers)):
            Z = np.dot(self.cache[f"A{i-1}"],
                       self.params[f"W{i}"]) + self.params[f"b{i}"]
            if i == len(self.layers) - 1:
                A = self.sigmoid(Z)
            else:
                A = self.relu(Z)
            self.cache[f"Z{i}"] = Z
            self.cache[f"A{i}"] = A
        return A

    def backward_propagation(self, y):
        grads = {}
        m = y.shape[0]
        L = len(self.layers) - 1
        dZ = self.cache[f"A{L}"] - y

        for i in reversed(range(1, L + 1)):
            grads[f"dW{i}"] = (1 / m) * np.dot(self.cache[f"A{i-1}"].T, dZ)
            grads[f"db{i}"] = (1 / m) * np.sum(dZ, axis=0, keepdims=True)
            if i > 1:
                dA = np.dot(dZ, self.params[f"W{i}"].T)
                dZ = dA * self.relu_derivative(self.cache[f"Z{i-1}"])

        return grads

    def update_parameters(self, grads):
        for i in range(1, len(self.layers)):
            self.params[f"W{i}"] -= self.lr * grads[f"dW{i}"]
            self.params[f"b{i}"] -= self.lr * grads[f"db{i}"]

    def fit(self, X, y):
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


def run_breast_cancer_classification():
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)

    X = X.fillna(X.mean())

    print("\nDataset Source: UCI Machine Learning Repository (via sklearn)")
    print("Dataset Name: Breast Cancer Dataset")
    print("Number of Samples:", X.shape[0])
    print("Number of Features:", X.shape[1])
    print("Problem Type: Binary Classification")
    print("Primary Metric: F1-score")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
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

    lr_model = LogisticRegressionScratch()
    lr_model.fit(X_train, y_train)
    mlp = MLPBinaryClassifier(layers=[X_train.shape[1], 32, 16, 1])
    mlp.fit(X_train, y_train)

    def compute_metrics(y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        TN, FP, FN, TP = cm.ravel()
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        precision = TP / (TP + FP + 1e-8)
        recall = TP / (TP + FN + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1, "TP": TP, "TN": TN, "FP": FP, "FN": FN}

    lr_pred = lr_model.predict(X_test)
    mlp_pred = mlp.predict(X_test)

    lr_metrics = compute_metrics(y_test, lr_pred)
    mlp_metrics = compute_metrics(y_test, mlp_pred)

    print("\n========== BREAST CANCER MODEL METRICS ==========")
    print("Logistic Regression:", {k: round(v, 4) for k, v in lr_metrics.items(
    ) if k in ["accuracy", "precision", "recall", "f1"]})
    print("MLP:", {k: round(v, 4) for k, v in mlp_metrics.items()
          if k in ["accuracy", "precision", "recall", "f1"]})

    plt.plot(lr_model.loss_history, label="Logistic Regression")
    plt.plot(mlp.loss_history, label="MLP")
    plt.legend()
    plt.title("Training Loss Comparison")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()

    labels = ["accuracy", "precision", "recall", "f1"]
    lr_vals = [lr_metrics[k] for k in labels]
    mlp_vals = [mlp_metrics[k] for k in labels]
    x = np.arange(len(labels))

    plt.bar(x - 0.2, lr_vals, 0.4, label="LR")
    plt.bar(x + 0.2, mlp_vals, 0.4, label="MLP")
    plt.xticks(x, labels)
    plt.legend()
    plt.title("Model Comparison")
    plt.show()

    f1_diff = mlp_metrics["f1"] - lr_metrics["f1"]
    print("\nANALYSIS")
    print(f"Logistic Regression F1-score: {lr_metrics['f1']:.4f}")
    print(f"MLP F1-score: {mlp_metrics['f1']:.4f}")
    if f1_diff > 0:
        print(
            f"MLP outperforms Logistic Regression by {f1_diff:.4f} in F1-score.")
    elif f1_diff < 0:
        print(
            f"Logistic Regression outperforms MLP by {-f1_diff:.4f} in F1-score.")
    else:
        print("Both models achieved identical F1-score.")
    print(
        f"Accuracy difference: {mlp_metrics['accuracy'] - lr_metrics['accuracy']:.4f}")
    print(f"Logistic Regression training time: {lr_model.training_time:.4f}s")
    print(f"MLP training time: {mlp.training_time:.4f}s")


# -----------------------------
# Task 3: Vehicle Multi-class Classification
# -----------------------------
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


class MLPMultiClassifier:
    def __init__(self, layers, lr=0.01):
        self.layers = layers
        self.lr = lr
        self.loss_history = []
        self.initialize_parameters()

    def initialize_parameters(self):
        self.W, self.b = [], []
        for i in range(len(self.layers) - 1):
            self.W.append(np.random.randn(
                self.layers[i], self.layers[i + 1]) * np.sqrt(2 / self.layers[i]))
            self.b.append(np.zeros((1, self.layers[i + 1])))

    def relu(self, z):
        return np.maximum(0, z)

    def relu_deriv(self, z):
        return (z > 0).astype(float)

    def softmax(self, z):
        z -= np.max(z, axis=1, keepdims=True)
        exp = np.exp(z)
        return exp / np.sum(exp, axis=1, keepdims=True)

    def forward_propagation(self, X):
        self.A, self.Z = [X], []
        for i in range(len(self.W) - 1):
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
                dZ = (dZ @ self.W[i].T) * self.relu_deriv(self.Z[i - 1])

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


def run_vehicle_classification():
    print("\nLoading dataset...")
    data = fetch_openml(name="vehicle", version=1, as_frame=True)
    X = data.data
    y = pd.factorize(data.target)[0]
    dataset_name = "Vehicle Dataset (Multi-class)"

    print("Dataset:", dataset_name)
    print("Samples:", X.shape[0])
    print("Features:", X.shape[1])
    print("Classes:", len(np.unique(y)))

    X = X.fillna(X.mean())
    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    num_classes = len(np.unique(y))
    y_train_ohe = np.zeros((len(y_train), num_classes))
    y_train_ohe[np.arange(len(y_train)), y_train] = 1

    softmax_model = SoftmaxRegression()
    mlp = MLPMultiClassifier([X_train.shape[1], 32, num_classes])

    start = time.time()
    softmax_model.fit(X_train, y_train_ohe)
    softmax_time = time.time() - start

    start = time.time()
    mlp.fit(X_train, y_train_ohe)
    mlp_time = time.time() - start

    def metrics(y_true, y_pred):
        return {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "Recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
            "F1 Score": f1_score(y_true, y_pred, average="macro", zero_division=0),
        }

    y_pred_softmax = softmax_model.predict(X_test)
    y_pred_mlp = mlp.predict(X_test)

    softmax_metrics = metrics(y_test, y_pred_softmax)
    mlp_metrics = metrics(y_test, y_pred_mlp)

    results_df = pd.DataFrame(
        [softmax_metrics, mlp_metrics], index=["Softmax", "MLP"])
    print("\n========== PERFORMANCE ==========")
    print(results_df.round(4).to_string())

    plt.figure()
    plt.plot(softmax_model.loss_history, label="Softmax")
    plt.plot(mlp.loss_history, label="MLP")
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
    plt.bar(x + 0.2, list(mlp_metrics.values()), 0.4, label="MLP")
    plt.xticks(x, labels)
    plt.ylabel("Score")
    plt.title("Model Comparison")
    plt.legend()
    plt.grid(axis="y")
    plt.show()

    f1_diff = mlp_metrics["F1 Score"] - softmax_metrics["F1 Score"]
    print("\nAnalysis:")
    print(f"MLP improves F1-score by {f1_diff:.4f}.")
    print(
        f"MLP Training Time: {mlp_time:.2f}s vs Softmax: {softmax_time:.2f}s.")
    print("MLP is computationally more expensive than Softmax.")


if __name__ == "__main__":
    run_house_prices_regression()
    run_breast_cancer_classification()
    run_vehicle_classification()
