import os
import json
from PIL import Image

# Загрузка аннотаций из COCO JSON-файла
def load_annotations(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

# Основная функция для вырезки и сохранения объектов
def crop_and_save_objects(annotations, images_info, images_dir):
    for ann in annotations['annotations']:
        # Получаем информацию о bounding box
        bbox = ann["bbox"]
        obj_type = ann["attributes"]["type"]

        # Получаем путь к изображению по image_id
        image_info = next(img for img in images_info if img['id'] == ann['image_id'])
        image_path = os.path.join(images_dir, image_info['file_name'])

        # Открываем изображение
        image = Image.open(image_path)

        # Создаем папку для объекта, если она не существует
        output_dir = os.path.join('images', obj_type)
        os.makedirs(output_dir, exist_ok=True)

        # Вырезаем объект по bbox (координаты bbox: x, y, ширина, высота)
        x, y, w, h = bbox
        if w > 100 and h > 100 :
            cropped_image = image.crop((x, y, x + w, y + h))
            file_name_without_extension = os.path.splitext(os.path.basename(image_path))[0]
            # Сохраняем вырезанное изображение с уникальным именем
            output_path = os.path.join(output_dir, f'{file_name_without_extension}_{ann["id"]}.jpg')
            cropped_image.save(output_path)

# Путь к JSON-аннотации и папке с изображениями
json_path = r'task_balka/annotations/instances_default.json'
images_dir = r'task_balka/images/'

# Загрузка аннотаций
annotations_data = load_annotations(json_path)
images_info = annotations_data['images']

# Вызов функции для обработки всех аннотаций
crop_and_save_objects(annotations_data, images_info, images_dir)
