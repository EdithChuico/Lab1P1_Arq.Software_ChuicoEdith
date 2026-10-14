class ValidateDatasetFilter:
    def __init__(self, target="rain"):
        self.target = target

    def run(self, dataframe):
        if dataframe.empty:
            raise ValueError("El dataset está vacío.")

        if self.target not in dataframe.columns:
            raise ValueError(
                f"No existe la columna objetivo '{self.target}'."
            )

        dataframe = dataframe.drop_duplicates().copy()
        
        dataframe = dataframe.dropna(subset=[self.target])

        if dataframe.empty:
            raise ValueError("No existen registros válidos.")

        dataframe[self.target] = (
            dataframe[self.target].astype(str).str.strip()
        )

        if dataframe[self.target].nunique() < 2:
            raise ValueError(
                "Se necesitan al menos dos clases."
            )

        return dataframe
