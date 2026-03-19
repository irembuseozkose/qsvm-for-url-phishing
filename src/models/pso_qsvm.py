import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold


class PSOQSVM:
    """
    PSO ile C hiperparametresini optimize eden QSVM modeli.
    Kernel dışarıdan verilir ve PSO içinde tekrar fit edilmez.
    """

    def __init__(
        self,
        quantum_kernel,
        n_particles=10,
        n_iters=20,
        C_min=1e-3,
        C_max=1e3,
        cv_splits=3,
        random_state=42
    ):
        # Kernel dışarıdan alınır
        self.kernel = quantum_kernel

        # PSO parametreleri
        self.n_particles = n_particles
        self.n_iters = n_iters
        self.C_min = C_min
        self.C_max = C_max
        self.cv_splits = cv_splits
        self.random_state = random_state

        # Sonuçlar
        self.best_C = None
        self.model = None
        self.X_train = None


    # ======================================
    # Kernel Matrislerini Cache’leme (Hız için)
    # ======================================
    def _precompute_kernel(self, X):
        """
        PSO süresince tekrar tekrar hesaplamamak için
        kernel önceden hesaplanıp hafızada tutulur.
        """
        n = len(X)
        K_full = self.kernel(X, X)
        return K_full


    # ======================================
    # Belirli bir C için CV ile doğruluk hesaplama
    # ======================================
    def _evaluate_C(self, C, K_full, y, splits):
        scores = []

        for train_idx, test_idx in splits:
            K_train = K_full[train_idx][:, train_idx]
            K_test  = K_full[test_idx][:, train_idx]

            clf = SVC(kernel="precomputed", C=C)
            clf.fit(K_train, y[train_idx])

            acc = clf.score(K_test, y[test_idx])
            scores.append(acc)

        return np.mean(scores)


    # ======================================
    # PSO Süreci
    # ======================================
    def _run_pso(self, X, y, K_full):
        rng = np.random.default_rng(self.random_state)

        # Stratified K-Fold (splitleri önceden hesapla)
        skf = StratifiedKFold(
            n_splits=self.cv_splits,
            shuffle=True,
            random_state=self.random_state
        )
        splits = list(skf.split(X, y))

        # Partikül başlangıcı
        particles = rng.uniform(self.C_min, self.C_max, self.n_particles)
        velocities = np.zeros(self.n_particles)

        # İlk skorlar
        p_best = particles.copy()
        p_best_scores = np.array([
            self._evaluate_C(c, K_full, y, splits) for c in particles
        ])

        # Global best
        g_best_idx = np.argmax(p_best_scores)
        g_best = p_best[g_best_idx]
        g_best_score = p_best_scores[g_best_idx]

        # PSO sabitleri
        w = 0.6
        c1 = 1.5
        c2 = 1.5

        # Iterate PSO
        for _ in range(self.n_iters):
            for i in range(self.n_particles):
                r1, r2 = rng.random(), rng.random()

                velocities[i] = (
                    w * velocities[i]
                    + c1 * r1 * (p_best[i] - particles[i])
                    + c2 * r2 * (g_best - particles[i])
                )

                particles[i] += velocities[i]
                particles[i] = np.clip(particles[i], self.C_min, self.C_max)

                score = self._evaluate_C(particles[i], K_full, y, splits)

                # Personal best update
                if score > p_best_scores[i]:
                    p_best[i] = particles[i]
                    p_best_scores[i] = score

                    # Global best update
                    if score > g_best_score:
                        g_best = particles[i]
                        g_best_score = score

        return g_best


    # ======================================
    # FIT
    # ======================================
    def fit(self, X, y):
        X = np.asarray(X)
        self.X_train = X

        # Kernel full precompute
        K_full = self._precompute_kernel(X)

        # PSO optimize → best C
        self.best_C = self._run_pso(X, y, K_full)
        print(f">> Best C (PSO): {self.best_C}")

        # Final model eğitimi
        self.model = SVC(kernel="precomputed", C=self.best_C)

        K_train = K_full
        self.model.fit(K_train, y)


    # ======================================
    # PREDICT & ACCURACY
    # ======================================
    def predict(self, X_test):
        X_test = np.asarray(X_test)
        K_test = self.kernel(X_test, self.X_train)
        return self.model.predict(K_test)

    def accuracy(self, X_test, y_test):
        return np.mean(self.predict(X_test) == y_test)
