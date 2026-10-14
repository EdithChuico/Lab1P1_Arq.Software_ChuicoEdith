import pandas as pd

class LoadDatasetFilter:
    def run(self, source):
        return pd.read_csv(source)
