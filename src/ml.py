from pickle import load
from catboost import CatBoostClassifier, Pool
import pandas as pd

def inference(data: list):
    if isinstance(data, list):
        data = pd.DataFrame(data, columns=['Popularity', 'danceability', 'energy', 'key', 'loudness', 'mode',
                                           'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence',
                                           'tempo', 'duration_in min/ms', 'time_signature'])
    with open("../models/v1/pipe.pcl", "rb") as fid:
        pipe = load(fid)

    model = CatBoostClassifier().load_model("../models/v1/model.pcl")
    x = pipe.transform(data)
    pool_inference = Pool(x)
    y_pr = model.predict(pool_inference)
    return y_pr


def main():
    test_sample = [[54.0, 0.382, 0.814, 3.0, -7.230, 1, 0.0406, 0.001100, 0.004010, 0.1010, 0.5690, 116.454, 251733.0, 4]]
    a = inference(test_sample)
    print(a)

if __name__ == "__main__":
    main()
