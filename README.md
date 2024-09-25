## Мульти-лейбл классификация дорожных фонарей
Модель: MobilenetV2<br>
Классы:
   1. Состояния фонаря:
      - Включен
      - Выключен
   2. Цвета свечения фонаря:
      - Белый 
      - Желтый
      - Фонарь выключен
   3. Тип фонаря:
      - Неопределен
      - K1
      - K2_1
      - K2_2
      - K3
      - K4

![alt text](other/streetligth_type.jpg)

### Настройка доступа к серверу

Указать username и путь к private_key в файле [Makefile](Makefile)

### Датасет

Скачать датасет (он окажется в папке dataset_coco):

```bash
make download_dataset
```

### Подготовка окружения

1. Создание и активация окружения
    ```bash
    python3 -m venv venv
    . venv/bin/activate 
   или
    . venv/Scripts/activate 
    ```

2. Установка библиотек
   ```
    make install
   ```
   
3. Запуск линтеров
   ```
   make lint
   ``` 

4. Логи в ClearML
- http://ml-server.avtodoria.ru:8080/projects/e89bf7670d654b87b9e251bb830ab08d/experiments/1a6ec9c072f64c96926b4cb58690b7b9/execution?columns=selected&columns=type&columns=name&columns=tags&columns=status&columns=project.name&columns=users&columns=started&columns=last_update&columns=last_iteration&columns=parent.name&order=-last_update&filter=
- http://ml-server.avtodoria.ru:8080/projects/e89bf7670d654b87b9e251bb830ab08d/experiments/e4ff653db01b4703bfb56d6485ccc7ba/execution?columns=selected&columns=type&columns=name&columns=tags&columns=status&columns=project.name&columns=users&columns=started&columns=last_update&columns=last_iteration&columns=parent.name&order=-last_update&filter=
5. Настраиваем [configs](configs) под себя.


### Обучение

Запуск тренировки:

```bash
make train
```

### Тест

Запуск тестирования:

```bash
make test
```

### Удаление логов

```bash
make clean-logs
```

### Инференс

Посмотреть результаты работы обученной сети можно в [тетрадке](notebooks/inference.ipynb).

