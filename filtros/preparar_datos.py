from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

class PrepareDataFilter:
    """Separa las variables y configura el preprocesamiento."""

    def __init__(self, target="rain"):
        self.target = target

    def run(self, dataframe):
        X = dataframe.drop(columns=[self.target]).copy()
        y = dataframe[self.target].copy()

        if X.shape[1] == 0:
            raise ValueError(
                "El dataset no contiene características predictoras."
            )

        numeric_columns = X.select_dtypes(
            include=["number", "bool"]
        ).columns.tolist()

        categorical_columns = [
            column for column in X.columns
            if column not in numeric_columns
        ]

        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median"))
        ])

        categorical_pipeline = Pipeline([
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore")
            )
        ])

        preprocessor = ColumnTransformer([
            ("numeric", numeric_pipeline, numeric_columns),
            (
                "categorical",
                categorical_pipeline,
                categorical_columns
            )
        ])

        info = {
            "target": self.target,
            "numeric_columns": numeric_columns,
            "categorical_columns": categorical_columns,
            "preprocessor": preprocessor
        }

        return X, y, info
