from sklearn.base import TransformerMixin

class Imputer(TransformerMixin):
    
    def __init__(self, strategy="mean", value=None):
        self._strategy = strategy
        self._value = value
    
    def fit(self, x):
        if self._strategy == "mean":
            self._value = x.mean()
        elif self._strategy == "mode":
            self._value = x.value_counts().index[0]
        else:
            raise ValueError("Unknow strategy {:s}".format(self._strategy))
        
        return self
        
    def transform(self, x):
        return x.fillna(self._value).to_frame()
    
    def get_params(self, *arg, **kwargs):
        return {"value": self._value, "strategy": self._strategy}