import numpy as np
from math import ceil, log2
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


class QuantumKernel:
    """
    Vektörize amplitude-encoding kernel.
    Orijinal QuantumKernel ile aynı sonucu verir, çok daha hızlıdır.
    """
    def __init__(self):
        self.n_features = None
        self.n_qubits   = None
        self.dim        = None

    def _init_from_data(self, X):
        self.n_features = X.shape[1]
        self.n_qubits   = ceil(log2(self.n_features))
        self.dim        = 2 ** self.n_qubits
        print(f">> FastQuantumKernel: {self.n_features} features → "
              f"{self.n_qubits} qubits (dim={self.dim})")

    def _encode_batch(self, X):
        """Tüm vektörleri tek seferde encode et — döngüsüz."""
        n = len(X)
        # Zero-pad to dim
        padded = np.zeros((n, self.dim))
        padded[:, :self.n_features] = X
        # L2 normalize (amplitude encoding gereği)
        norms = np.linalg.norm(padded, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return padded / norms

    def __call__(self, X1, X2):
        X1 = np.asarray(X1, dtype=float)
        X2 = np.asarray(X2, dtype=float)

        if self.n_features is None:
            self._init_from_data(X1)

        S1 = self._encode_batch(X1)  # (n1, dim)
        S2 = self._encode_batch(X2)  # (n2, dim)

        # Kernel = |<ψ_i|ψ_j>|² = (S1 @ S2.T)²  — tek matris çarpımı
        inner = S1 @ S2.T
        return inner ** 2

    def __repr__(self):
        return (f"FastQuantumKernel(n_features={self.n_features}, "
                f"n_qubits={self.n_qubits})")