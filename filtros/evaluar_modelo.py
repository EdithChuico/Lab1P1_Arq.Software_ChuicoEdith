from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

class EvaluateModelFilter:
    """Evalúa el modelo con los datos de prueba."""

    def run(self, model, X_test, y_test):
        predictions = model.predict(X_test)
        labels = list(model.classes_)

        metrics = {
            "accuracy": round(
                accuracy_score(y_test, predictions) * 100, 2
            ),
            "precision": round(
                precision_score(
                    y_test, predictions,
                    average="weighted",
                    zero_division=0
                ) * 100, 2
            ),
            "recall": round(
                recall_score(
                    y_test, predictions,
                    average="weighted",
                    zero_division=0
                ) * 100, 2
            ),
            "f1": round(
                f1_score(
                    y_test, predictions,
                    average="weighted",
                    zero_division=0
                ) * 100, 2
            ),
            "labels": labels,
            "confusion_matrix": confusion_matrix(
                y_test,
                predictions,
                labels=labels
            ).tolist()
        }

        return metrics
