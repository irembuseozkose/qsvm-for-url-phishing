from sklearn.svm import SVC
import numpy as np

class QSVM:
    def __init__(self, quantum_kernel, C=1.0):
        self.kernel = quantum_kernel
        self.model = SVC(kernel="precomputed")
        self.X_train = None

    def fit(self, X_train, y_train):
        self.X_train = np.asarray(X_train)

        # Kernel matrisi (train-train)
        K_train = self.kernel(self.X_train, self.X_train)

        # Fit SVM
        self.model.fit(K_train, y_train)

    def predict(self, X_test):
        # Kernel matrisi (test-train)
        K_test = self.kernel(np.asarray(X_test), self.X_train)
        return self.model.predict(K_test)

    def accuracy(self, X_test, y_test):
        y_pred = self.predict(X_test)
        return float(np.mean(y_pred == y_test))
