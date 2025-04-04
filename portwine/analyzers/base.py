class Analyzer:
    def analyze(self, results, *args, **kwargs):
        # Analyze method. Should return some sort of dataframe, etc
        raise NotImplementedError

    def plot(self, results, *args, **kwargs):
        # Optional visualization method
        raise NotImplementedError
