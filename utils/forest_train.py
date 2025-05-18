import os
import joblib
import re

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

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
    
    X = df.drop(["class", "image_name", "crop_name", "image_width", "image_height"], axis=1)
    print(X.columns)
    y = df["class"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = joblib.load(conf.BBOX_CLASSIFIER_PATH)
    # model = RandomForestClassifier(n_estimators=50, random_state=42)
    # model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-score: {f1}")
    
    print("===============")
    y_pred = model.predict(X)

    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, average='weighted')
    recall = recall_score(y, y_pred, average='weighted')
    f1 = f1_score(y, y_pred, average='weighted')
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-score: {f1}")
    
    feature_importances = model.feature_importances_

    feature_importance_df = pd.DataFrame({"Feature": X.columns, "Importance": feature_importances})
    feature_importance_df = feature_importance_df.sort_values(by="Importance", ascending=False)

    print("Importances:")
    print(feature_importance_df)

    joblib.dump(model, "tmp/boxes_classifier.pkl")
    print("Model saved successfully.")
    

    '''
    cols_from_prev_df = ["x", "y", "x1", "x2", "y1", "y2", "area", "height/width", "width/height", "crop_height", "crop_height_%", "crop_width", "crop_width_%"]
    df2 = {"name": []}
    for i in range(conf.NUM_CLUSSES):
        df2[f"class_{i}"] = []
        for elem in cols_from_prev_df:
            df2[f"{elem}_{i}"] = []
        for j in range(conf.NUM_CLUSSES):
            if i == j:
                continue
            df2[f"{i}_to_{j}_x"] = []
            df2[f"{i}_to_{j}_y"] = []
    df2["good/bad"] = []

    for image_name in df["image_name"].unique():
        local_df = df[df["image_name"] == image_name]

        df2["name"].append(image_name)
        if os.path.exists(path):
            df2["good/bad"].append(0)
        else:
            df2["good/bad"].append(1)
        for i in range(conf.NUM_CLUSSES):
            if local_df[local_df["class"] == i].shape[0] == 0:
                df2[f"class_{i}"].append(0)
                for elem in cols_from_prev_df:
                    df2[f"{elem}_{i}"].append(0)
            else:
                df2[f"class_{i}"].append(1)
                for elem in cols_from_prev_df:
                    df2[f"{elem}_{i}"].append(float(local_df[local_df["class"] == i][elem].iloc[0]))
            for j in range(conf.NUM_CLUSSES):
                if i == j:
                    continue
                if local_df[local_df["class"] == i].shape[0] == 0 or local_df[local_df["class"] == j].shape[0] == 0:
                    df2[f"{i}_to_{j}_x"].append(0)
                    df2[f"{i}_to_{j}_y"].append(0)
                else:
                    vec_x = float(local_df[local_df["class"] == j]["x"].iloc[0]) - float(local_df[local_df["class"] == i]["x"].iloc[0])
                    vec_y = float(local_df[local_df["class"] == j]["y"].iloc[0]) - float(local_df[local_df["class"] == i]["y"].iloc[0])
                    df2[f"{i}_to_{j}_x"].append(vec_x)
                    df2[f"{i}_to_{j}_y"].append(vec_y)

            
    df2 = pd.DataFrame(df2)
    df2.to_csv("tmp/tmp.csv")
    '''
    
    df2 = pd.read_csv("tmp/tmp.csv")
    
    print(len(df2.columns))

    X = df2.drop(["name", "good/bad", "Unnamed: 0"], axis=1)
    print(X.columns)
    y = df2["good/bad"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # model = joblib.load(conf.PATTERN_CLASSIFIER_PATH)
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-score: {f1}")
    
    print("===============")
    y_pred = model.predict(X)

    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, average='weighted')
    recall = recall_score(y, y_pred, average='weighted')
    f1 = f1_score(y, y_pred, average='weighted')
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-score: {f1}")
    
    feature_importances = model.feature_importances_

    feature_importance_df2 = pd.DataFrame({"Feature": X.columns, "Importance": feature_importances})
    feature_importance_df2 = feature_importance_df2.sort_values(by="Importance", ascending=False)

    print("Importances:")
    print(feature_importance_df2)
    
    print(feature_importance_df2[feature_importance_df2["Importance"] > 0.0])

    joblib.dump(model, "tmp/pattern_classifier.pkl")
    print("Model saved successfully.")


if __name__ == "__main__":
    main("local_data/crops.csv")
