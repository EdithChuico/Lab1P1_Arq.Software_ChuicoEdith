from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

class TrainModelFilter:
    """Construye y entrena el modelo de clasificación."""

    def run(self, X_train, y_train, info):
        model = Pipeline([
            ("preprocessor", info["preprocessor"]),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=100,
                    random_state=42,
                    class_weight="balanced"
                )
            )
        ])

        model.fit(X_train, y_train)

        return model
