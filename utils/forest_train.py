import os

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib 

import src.conf as conf


def main(csv_path: str, num_classes: int = conf.NUM_CLUSSES) -> None:
    df = pd.read_csv(csv_path)
    
    print(num_classes)
    df = df[df["class"] < num_classes]
    
    center_x = df["image_width"] / 2
    center_y = df["image_height"] / 2

    df["x1"] -= center_x
    df["x2"] -= center_x
    df["y1"] -= center_y
    df["y2"] -= center_y

    df["crop_width"] = df["x2"] - df["x1"]
    df["crop_width_%"] = df["crop_width"] / df["image_width"]
    df["crop_height"] = df["y2"] - df["y1"]
    df["crop_height_%"] = df["crop_height"] / df["image_height"]
    df["x"] = (df["x2"] + df["x1"]) / 2
    df["y"] = (df["y2"] + df["y1"]) / 2
    df["area"] = df["crop_height"] * df["crop_width"]
    df["width/height"] = df["crop_width"] / df["crop_height"]
    df["height/width"] = df["crop_height"] / df["crop_width"]
    
    image_params = {}
    for i in range(len(df)):
        path = os,path.join("local_data", "output", "Bad", df["crop_name"][i])
        if not os.path.exists(path):
            continue
        path = os,path.join("local_data", "output", "Good", df["crop_name"][i])
        if not os.path.exists(path):
            continue
        img = cv2

    df = df.drop(["image_name", "crop_name", "image_width", "image_height"], axis=1)
    
    X = df.drop(["class"], axis=1)
    print(X.columns)
    y = df["class"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
    
    feature_importances = model.feature_importances_

    feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    print("Importances:")
    print(feature_importance_df)

    joblib.dump(model, "forest_model.pkl")
    print("Model saved successfully.")


if __name__ == "__main__":
    main("local_data/crops.csv")
