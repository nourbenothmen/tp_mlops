from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pickle

def test_model_accuracy():
    iris = load_iris()
    X, y = iris.data, iris.target
    with open("model.pkl", "rb") as f:  # Changed from ../model.pkl to model.pkl
        model = pickle.load(f)
    acc = model.score(X, y)
    assert acc > 0.9, "La précision du modèle est inférieure à 90%"