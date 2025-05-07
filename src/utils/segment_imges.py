import glob


def process_images_in_folder(model, folder_path: str):
    image_paths = (glob.glob(os.path.join(folder_path, '*.jpg')) + glob.glob(os.path.join(folder_path, '*.png')) 
                + glob.glob(os.path.join(folder_path, '*.jpeg')) + glob.glob(os.path.join(folder_path, '*.PNG')))
    
    for image_path in image_paths:
        print(f"Обрабатываем изображение: {image_path}")
        model.predict_image(image_path)
        model.calc_masks()
        print(f"Изображение {image_path} обработано.")