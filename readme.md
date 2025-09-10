# TrioLearn: From Scratch Supervised Classification ML Library

**TrioLearn** is a lightweight, from-scratch Python library for supervised machine learning classification. This is a **side project**, still under active development. Features are gradually being added, and the library currently supports binary classification with plans for multi-class in the future.

---

## ⚠️ Note

* The library is **under development** and not fully complete.
* Bugs, issues, and suggestions are **welcome**! Feel free to reach out if you have ideas or contributions.

---

## 🚧 Current Constraints

To use the library effectively, your dataset must follow these rules:

1. The **target/output column must be the last column**.
2. **Categorical columns must not be pre-encoded** to numerical values.

---

## 🛠 Library Components

### 1. Data Preprocessing

The `DataPreprocessing` module prepares raw datasets for machine learning. It performs:

* **Normalization**: Numerical columns with a range > 50 are normalized.
* **One-Hot Encoding**: Converts categorical columns into numeric format.
* **Missing Value Handling**: Replaces missing values with **mean** (numerical) or **mode** (categorical).
* **Dataset Splitting**: Separates the dataset into **features (X)** and **target (Y)**.

> **Note:** Preprocessing is optional if your dataset is already clean and in the required format.

---

### 2. Machine Learning Algorithms

After preprocessing, you can use any of the supported algorithms:

1. **Binary Logistic Regressor**
2. **Naive Bayes Classifier (for mixed data)**
3. **Classification Tree**
4. **Artificial Neural Network (Multilayer Perceptron)**
5. **K-Nearest Neighbour (KNN)**

> Currently, most algorithms are **binary classification only**. Multi-class support is planned for future releases.

---

## 💻 Usage Example

```python
from sklearn.model_selection import train_test_split
import pandas as pd

from triolearn.data_preprocessing import DataPrepocessing
from triolearn.machine_learning.MLP import NeuralNetwork
from triolearn.machine_learning.Probabillistic import NaiveBayes
from triolearn.machine_learning.Regression import Binary_Logistic_Regressor
from triolearn.machine_learning.NearestNeighbor import KNN

# Load dataset
data = pd.read_csv('dataset.csv')
train_dataframe =  data.iloc[:int(len(data)*0.8), :]
test_dataframe  =  data.iloc[int(len(data)*0.8):, :]

# Preprocess data
X, y = DataPrepocessing().dataCleaning(data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)

# Initialize models
NN_clf  = NeuralNetwork([64, 32, 16, 8], epochs=10)
NB_clf  = NaiveBayes()
LR_clf  = Binary_Logistic_Regressor(
    main_gradient_descent='mini-batch',
    regularizer='l2',
    hyper_parameters_assign='auto-assign',
    hyper_parameters_tuning_gradient_descent='mini-batch',
    max_iter=1000,
    early_stopping=True,
    monitor='val_accuracy',
    paitiance=3,
    error_roundoff=8,
    acc_roundoff=4,
    acc_change=1e-3,
    error_change=1e-7,
    verbose=True
)
KNN_clf = KNN()

# Training
print("Training Models...")
NN_clf.fit(X_train, y_train)
# NB_clf.fit(train_dataframe)  # Optional
LR_clf.fit(X_train, y_train, X_test, y_test)
KNN_clf.fit(X_train, y_train)

# Evaluation
print("Model Evaluation:")
print("Neural Network Accuracy:", NN_clf.score(X_test, y_test))
print("Naive Bayes Accuracy:", NB_clf.score(test_dataframe.iloc[:, -1], test_dataframe))
print("Logistic Regression Accuracy:", LR_clf.score)
print("KNN Accuracy:", KNN_clf.score(X_test, y_test))
```

---

## 📌 Future Improvements

* Multi-class classification support
* More optimized and efficient algorithms
* Enhanced preprocessing features
* Additional evaluation metrics

---

## 🤝 Contributing

Any contributions, suggestions, or bug reports are welcome! Contact me via my profile for collaboration or to provide feedback.

---

## ⚡ Summary

TrioLearn is a **from-scratch supervised ML library** aimed at learning and experimentation. It offers **easy-to-use interfaces** for binary classification algorithms and is continuously evolving.

