from pathlib import Path

import joblib
import pandas as pd

from flask import Flask, render_template, request, flash
from sklearn.model_selection import train_test_split

from filtros import (
    LoadDatasetFilter,
    ValidateDatasetFilter,
    PrepareDataFilter,
    TrainModelFilter,
    EvaluateModelFilter
)

# Configuración de Flask
app = Flask(
    __name__,
    static_folder="public",
    static_url_path="/public"
)

app.secret_key = "cambia-esta-clave-en-produccion"

# Configuración del modelo
MODEL_PATH = Path("models/weather_model.joblib")
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

TARGET = "rain"

@app.route("/", methods=["GET"])
def index():
    return render_template(
        "index.html",
        results=None,
        prediction_features=None,
        prediction_result=None
    )

@app.route("/train", methods=["POST"])
def train():
    uploaded = request.files.get("dataset")

    if uploaded is None or not uploaded.filename:
        flash("Selecciona un archivo CSV.")
        return index()

    if not uploaded.filename.lower().endswith(".csv"):
        flash("El archivo debe tener extensión .csv.")
        return index()

    try:
        # Filtro 1: carga del dataset
        raw = LoadDatasetFilter().run(uploaded.stream)

        # Filtro 2: validación
        validated = ValidateDatasetFilter(TARGET).run(raw)

        # Filtro 3: preparación de los datos
        X, y, info = PrepareDataFilter(TARGET).run(validated)

        # Separación de datos de entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y
        )

        # Filtro 4: entrenamiento
        model = TrainModelFilter().run(
            X_train,
            y_train,
            info
        )

        # Filtro 5: evaluación
        metrics = EvaluateModelFilter().run(
            model,
            X_test,
            y_test
        )

        # Guardar modelo y configuración
        joblib.dump(
            {
                "model": model,
                "features": list(X.columns),
                "target": info["target"]
            },
            MODEL_PATH
        )

        results = {
            "rows": len(validated),
            "features": len(X.columns),
            "target": info["target"],
            "class_distribution": {
                str(key): int(value)
                for key, value in y.value_counts().items()
            },
            "metrics": metrics,
            "model_path": str(MODEL_PATH)
        }

        return render_template(
            "index.html",
            results=results,
            prediction_features=list(X.columns),
            prediction_result=None
        )

    except Exception as exc:
        app.logger.exception("Error durante el entrenamiento")
        flash(f"No se pudo procesar el dataset: {exc}")
        return index()

@app.route("/predict", methods=["POST"])
def predict():
    if not MODEL_PATH.exists():
        flash("Primero entrena un modelo subiendo un dataset.")
        return index()

    try:
        saved = joblib.load(MODEL_PATH)

        features = saved["features"]
        model = saved["model"]

        values = {}

        for feature in features:
            raw_value = request.form.get(feature, "").strip()

            if raw_value == "":
                raise ValueError(
                    f"El campo '{feature}' es obligatorio."
                )

            # Convertir a número cuando corresponda.
            try:
                values[feature] = float(raw_value)
            except ValueError:
                values[feature] = raw_value

        sample = pd.DataFrame(
            [values],
            columns=features
        )

        prediction = model.predict(sample)[0]

        probability = None

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(sample)[0]
            probability = round(
                float(max(probabilities)) * 100,
                2
            )

        prediction_result = {
            "label": str(prediction),
            "probability": probability
        }

        return render_template(
            "index.html",
            results=None,
            prediction_features=features,
            prediction_result=prediction_result
        )

    except Exception as exc:
        app.logger.exception("Error durante la predicción")
        flash(f"No se pudo generar la predicción: {exc}")
        return index()

if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5001,
        debug=True
    )
