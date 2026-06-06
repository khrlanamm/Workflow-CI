import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import mlflow

script_dir = os.path.dirname(os.path.abspath(__file__))
x_path = os.path.join(script_dir, 'dataset_preprocessing/X_processed.csv')
y_path = os.path.join(script_dir, 'dataset_preprocessing/y_processed.csv')
X = pd.read_csv(x_path)
y = pd.read_csv(y_path).values.ravel()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
with mlflow.start_run():
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(clf, "model")
