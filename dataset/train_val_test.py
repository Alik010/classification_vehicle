import os
import json
import random
from sklearn.model_selection import train_test_split

# Путь к папкам с изображениями (где каждая папка названа по type)
data_dir = r'C:\Users\aliko\PycharmProjects\cls_vehicle\dataset\images'


# Функция для разделения файлов на train, val, test
def split_data(files, train_size=0.8, val_size=0.1, test_size=0.1):
    # Проверяем, что сумма пропорций равна 1
    assert train_size + val_size + test_size == 1.0, "Размеры выборок должны в сумме давать 1"

    # Разделяем данные на train и временный набор (для дальнейшего разделения на val и test)
    train_files, temp_files = train_test_split(files, train_size=train_size, random_state=42)

    # Оставшиеся данные разделяем на val и test
    val_size_adjusted = val_size / (val_size + test_size)  # корректируем размер в зависимости от остатка
    val_files, test_files = train_test_split(temp_files, train_size=val_size_adjusted, random_state=42)

    return train_files, val_files, test_files


# Инициализируем структуры для хранения всех данных
all_train = []
all_val = []
all_test = []

# Проходим по каждой папке (типу объекта)
for obj_type in os.listdir(data_dir):
    folder_path = os.path.join(data_dir, obj_type)

    if os.path.isdir(folder_path):
        # Получаем список всех файлов в папке
        files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

        # Разделяем файлы на train, val, test
        train_files, val_files, test_files = split_data(files)

        # Добавляем файлы в соответствующие списки
        all_train.extend([{"file_name": f, "type": obj_type} for f in train_files])
        all_val.extend([{"file_name": f, "type": obj_type} for f in val_files])
        all_test.extend([{"file_name": f, "type": obj_type} for f in test_files])

# Перемешиваем данные
random.shuffle(all_train)
random.shuffle(all_val)
random.shuffle(all_test)

path = "C:/Users/aliko/PycharmProjects/cls_vehicle/dataset/"
# Сохранение train, val, test в отдельные JSON-файлы
with open(f'{path}annotations/train.json', 'w') as f:
    json.dump(all_train, f, indent=4)

with open(f'{path}annotations/val.json', 'w') as f:
    json.dump(all_val, f, indent=4)

with open(f'{path}annotations/test.json', 'w') as f:
    json.dump(all_test, f, indent=4)

print(f"Данные разделены и сохранены в файлы: train.json, val.json, test.json")
