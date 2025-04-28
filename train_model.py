 
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Chargement des données
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

# Séparation des données
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialisation de MLflow
mlflow.set_experiment("iris-classification")

# Début de l'expérience
with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)

    # Enregistrement des paramètres et métriques
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(model, "random_forest_model")
    print(f"Modèle entraîné avec une précision de {acc:.4f}")

    # Sauvegarde du modèle
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)