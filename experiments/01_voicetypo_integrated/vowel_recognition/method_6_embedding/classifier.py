"""임베딩 기반 모음 분류기. SVM/MLP/LR 지원."""

import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

VOWELS = ["아", "어", "오", "우", "으", "이", "에", "애"]


class EmbeddingVowelClassifier:
    def __init__(self, classifier_type='svm'):
        self._type = classifier_type
        self._scaler = StandardScaler()
        self._clf = None
        self._trained = False

    @property
    def is_trained(self):
        return self._trained

    def train(self, X: np.ndarray, y: np.ndarray):
        """학습. X: (N, 768), y: (N,) 문자열 라벨."""
        X_scaled = self._scaler.fit_transform(X)

        if self._type == 'svm':
            self._clf = SVC(kernel='rbf', C=10.0, gamma='scale',
                            probability=True, random_state=42)
        elif self._type == 'mlp':
            self._clf = MLPClassifier(
                hidden_layer_sizes=(128,), max_iter=500,
                alpha=0.01, random_state=42)
        elif self._type == 'lr':
            self._clf = LogisticRegression(
                C=1.0, max_iter=1000,
                multi_class='multinomial', random_state=42)
        else:
            raise ValueError(f"Unknown classifier: {self._type}")

        self._clf.fit(X_scaled, y)
        self._trained = True

    def predict(self, X: np.ndarray):
        """예측. X: (N, 768) 또는 (768,).
        Returns: (labels, confidences)
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)
        X_scaled = self._scaler.transform(X)
        labels = self._clf.predict(X_scaled)
        proba = self._clf.predict_proba(X_scaled)
        confidences = proba.max(axis=1)
        return labels, confidences

    def predict_one(self, embedding: np.ndarray):
        """단일 임베딩 → (모음, 신뢰도, 확률dict)."""
        X = embedding.reshape(1, -1)
        X_scaled = self._scaler.transform(X)
        label = self._clf.predict(X_scaled)[0]
        proba = self._clf.predict_proba(X_scaled)[0]
        classes = self._clf.classes_
        prob_dict = {cls: float(p) for cls, p in zip(classes, proba)}
        conf = max(proba)
        return label, conf, prob_dict

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({
                'type': self._type,
                'scaler': self._scaler,
                'clf': self._clf,
            }, f)

    def load(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self._type = data['type']
        self._scaler = data['scaler']
        self._clf = data['clf']
        self._trained = True
