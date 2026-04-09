import numpy as np
from math import ceil, log2
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


class QuantumKernel:
    """
    Amplitude-encoding tabanlı quantum kernel.
    fit() gerekmez — __call__(X1, X2) ile direkt kullanılır.
    İlk çağrıda feature sayısını X1'den otomatik öğrenir.
    """

    def __init__(self):
        self.n_features = None
        self.n_qubits   = None
        self.dim        = None

    def _init_from_data(self, X: np.ndarray):
        self.n_features = X.shape[1]
        self.n_qubits   = ceil(log2(self.n_features))
        self.dim        = 2 ** self.n_qubits
        print(
            f">> QuantumKernel: "
            f"{self.n_features} features → {self.n_qubits} qubits (dim={self.dim})"
        )

    def _encode_state(self, x: np.ndarray) -> Statevector:
        vec = np.zeros(self.dim, dtype=float)
        vec[: self.n_features] = x
        norm = np.linalg.norm(vec)
        if norm == 0:
            raise ValueError("Cannot amplitude encode zero vector.")
        vec /= norm
        qc = QuantumCircuit(self.n_qubits)
        qc.initialize(vec, range(self.n_qubits))
        return Statevector.from_instruction(qc)

    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        X1 = np.asarray(X1, dtype=float)
        X2 = np.asarray(X2, dtype=float)

        # İlk çağrıda otomatik initialize
        if self.n_features is None:
            self._init_from_data(X1)

        states1 = [self._encode_state(x) for x in X1]
        states2 = [self._encode_state(x) for x in X2]

        K = np.zeros((len(states1), len(states2)))
        for i, s1 in enumerate(states1):
            for j, s2 in enumerate(states2):
                K[i, j] = np.abs(np.vdot(s1.data, s2.data)) ** 2

        return K

    def __repr__(self):
        return (
            f"QuantumKernel(encoding=amplitude, "
            f"n_features={self.n_features}, "
            f"n_qubits={self.n_qubits})"
        )