import json
import pandas as pd


def main(json_path: str, csv_save_path: str) -> None:
    with open(json_path, "r") as file:
        data = json.load(file)
    
    data_csv = {
        "image_name": [],
        "image_height": [],
        "image_width": [],
        "crop_name": [],
        "x1": [],
        "y1": [],
        "x2": [],
        "y2": [],
        "class": [],
    }
    for elem in data:
        num_crops = len(elem["crops"])
        data_csv["image_name"].extend([elem["image_name"]] * num_crops)
        data_csv["image_width"].extend([elem["image_size"][0]] * num_crops)
        data_csv["image_height"].extend([elem["image_size"][1]] * num_crops)
        for crop in elem["crops"]:
            data_csv["crop_name"].append(crop["crop_name"])
            data_csv["x1"].append(crop["bbox"][0])
            data_csv["y1"].append(crop["bbox"][1])
            data_csv["x2"].append(crop["bbox"][2])
            data_csv["y2"].append(crop["bbox"][3])
            data_csv["class"].append(crop["class"])
    
    df = pd.DataFrame(data_csv)
    
    df.to_csv(csv_save_path, index=False)


if __name__ == "__main__":
    main("local_data/output/crops_data.json", "tmp/crops.csv")
