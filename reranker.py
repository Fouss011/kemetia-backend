import numpy as np
from sklearn.linear_model import LogisticRegression

class MiniReranker:
    """
    Petit modèle qui apprend à pondérer:
      - score de similarité d'embedding (cosine)
      - bonus de catégorie (ex: salutations, santé, etc. si utile)
      - longueur de la requête
    Très simple mais suffisant pour corriger les priorités au fil des retours.
    """
    def __init__(self):
        self.clf = None

    def fit(self, rows, events, features_per_row):
        """
        rows: dict row_id -> row (audio_meta)
        events: liste d'events {row_id, accepted}
        features_per_row: dict row_id -> feature vector np.array([...])
        """
        X, y = [], []
        for ev in events:
            rid = ev.get("row_id")
            if rid not in features_per_row:
                continue
            X.append(features_per_row[rid])
            y.append(1 if ev.get("accepted") else 0)
        if not X:
            self.clf = None
            return
        X = np.vstack(X)
        y = np.array(y)
        self.clf = LogisticRegression(max_iter=500).fit(X, y)

    def predict_bonus(self, feat_vec):
        if self.clf is None:
            return 0.0
        # probabilité d'acceptation -> petit bonus [-0.1, +0.1]
        p = float(self.clf.predict_proba(feat_vec.reshape(1, -1))[0,1])
        return (p - 0.5) * 0.2  # range approx [-0.1..+0.1]
