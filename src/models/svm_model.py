# svm_model.py
from sklearn.svm import SVC
import numpy as np


class ClassicalSVM:

    def __init__(self, C=1.0, kernel="rbf", gamma="scale"):
        self.model = SVC(C=C, kernel=kernel, gamma=gamma)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def accuracy(self, X_test, y_test):
        pred = self.predict(X_test)
        return float(np.mean(pred == y_test))
