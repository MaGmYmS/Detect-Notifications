import json
import os
import random
import shutil

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from roboflow import Roboflow
from ultralytics import YOLO

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)


class CustomYOLOv8Model:
    # region чтение датасета
    @staticmethod
    def _update_data_yaml(data_yaml_path):
        """
        Обновляет файл данных YAML для указания путей к изображениям обучения и валидации.

        :param data_yaml_path: Путь к файлу данных YAML.
        :type data_yaml_path: str
        :return: Нет возвращаемого значения.
        """
        train_path = "../train/images"
        val_path = "../valid/images"

        with open(data_yaml_path, 'r') as file:
            data = yaml.safe_load(file)

        data['train'] = train_path
        data['val'] = val_path

        with open(data_yaml_path, 'w') as file:
            yaml.dump(data, file)

    def download_dataset(self):
        """
        Загружает датасет с Roboflow и организует его структуру.

        Качает датасет с использованием API-ключа Roboflow. Перемещает файлы из исходной папки в целевую,
        а затем обновляет файл данных YAML с новыми путями.

        :return: Нет возвращаемого значения.
        """

        # качаем датасет с Roboflow
        rf = Roboflow(api_key="AmQ0vHqaiNHr6SeXUAWb")
        project = rf.workspace("detected-notification").project("detected_notification")
        dataset = project.version(3).download("yolov8")

        # Путь к исходной папке
        source_folder = dataset.location
        dataset_name = f"{dataset.name}-{dataset.version}"
        # Путь к целевой папке
        target_folder = os.path.join("yolov5", "datasets", dataset_name)

        # Если папка не существует, создайте ее
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)

        # Переместите файлы в целевую папку
        for file_name in os.listdir(source_folder):
            source_file_path = os.path.join(source_folder, file_name)
            target_file_path = os.path.join(target_folder, file_name)
            print(f"Move file {file_name}")
            shutil.move(source_file_path, target_file_path)

        project = rf.workspace("detected-notification").project("detected_notification")
        project.version(3).download("yolov8")

        data_yaml_path = f"{dataset_name}/data.yaml"
        self._update_data_yaml(data_yaml_path)

    # endregion

    @staticmethod
    def train_my_model():
        """
        Обучает модель YOLOv8 на предоставленных данных.

        Инициализирует и обучает модель YOLOv8 на указанных данных.
        В данном случае, обучение происходит на протяжении 60 эпох с размером изображения 640x640.

        :return: Нет возвращаемого значения.
        """

        name_model = "yolov8m.pt"
        model = YOLO(name_model)
        model.train(data="Detected_Notification-3/data.yaml", epochs=60, imgsz=640)

    @staticmethod
    def _plot_results(coordinates, categories, image_path):
        """
        Визуализирует результаты детекции объектов на изображении.

        Использует переданные координаты, категории объектов и путь к изображению для построения графического
        представления результатов детекции.

        :param coordinates: Список координат объектов в формате (x, y, x_max, y_max).
        :type coordinates: list of tuples
        :param categories: Список категорий объектов.
        :type categories: list
        :param image_path: Путь к изображению.
        :type image_path: str
        :return: Нет возвращаемого значения.
        """

        image = plt.imread(image_path)

        fig, ax = plt.subplots(1)
        ax.imshow(image)

        colors = {0: 'r', 1: 'lime', 2: 'skyblue'}
        labels = {0: 'Button', 1: 'Site', 2: 'System'}

        for coord, category in zip(coordinates, categories):
            x, y, x_max, y_max = coord

            color = colors.get(category, 'black')  # Если категория не известна, используется черный цвет
            label = labels.get(category, 'Unknown')

            rect = patches.Rectangle((x, y), x_max - x, y_max - y, linewidth=1, edgecolor=color, facecolor='none',
                                     label=label)
            ax.add_patch(rect)

            ax.text(x, y, label, color=color, fontsize=8, verticalalignment='bottom')

        plt.show()

    # region формирование результатов
    @staticmethod
    def _determine_object_type(class_id):
        """
        Определяет тип объекта на основе его класса.

        :param class_id: Идентификатор класса объекта.
        :type class_id: int
        :return: Строка, представляющая тип объекта (например, "system", "site", "unknown").
        :rtype: str
        """

        # Функция для определения типа объекта по его классу
        if class_id == 2:
            return "system"
        elif class_id == 1:
            return "site"
        else:
            return "unknown"

    def _infer_objects(self, results):
        """
        Выполняет вывод информации о распознанных объектах и сохраняет результаты в JSON-файл.

        :param results: Результаты работы модели.
        :type results: object
        :return: Строка JSON с информацией о распознанных объектах, координаты и категории объектов.
        :rtype: str
        """

        # Получение координат распознанных объектов
        coordinates = results[0].boxes.xyxy.cpu().numpy()
        categories = results[0].boxes.cls.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()

        detected_objects = []

        for i, coord in enumerate(coordinates):  # добавляем уведомления (site/system)
            target_int = int(categories[i])
            target = self._determine_object_type(target_int)
            if target_int != 0:
                detected_object = {
                    "order": "неизвестен",
                    "type": target,
                    "confidence": float(confidences[i]),
                    "coordinate": {
                        "top_left": {
                            "x": int(coord[0]),
                            "y": int(coord[1])
                        },
                        "bottom_right": {
                            "x": int(coord[2]),
                            "y": int(coord[3]),
                        }
                    },
                    "buttons": []
                }
                detected_objects.append(detected_object)

        for i, coord in enumerate(coordinates):  # добавляем кнопки
            target_int = int(categories[i])
            if target_int == 0:  # если кнопка, то...
                button_data = {
                    "name": "неизвестно",
                    "content": "неизвестно",
                    "confidence": float(confidences[i]),  # Уверенность модели
                    "coordinate": {
                        "top_left": {
                            "x": int(coord[0]),
                            "y": int(coord[1])
                        },
                        "bottom_right": {
                            "x": int(coord[2]),
                            "y": int(coord[3]),
                        }
                    },
                }

                for tmp_i in range(len(detected_objects)):
                    # Проверка, что кнопка находится внутри уведомления
                    if self._is_inside_notification(button_data, detected_objects[tmp_i]):
                        detected_objects[tmp_i]["buttons"].append(button_data)

        result_dict = {"detected_notifications": detected_objects}

        # Сериализация в JSON и запись в файл
        with open('output.json', 'w') as json_file:
            json.dump(result_dict, json_file, indent=4)
        json_response = json.dumps(result_dict, indent=4)

        return json_response, coordinates, categories

    @staticmethod
    def _is_inside_notification(button_data, notification_data):
        """
        Проверяет, находится ли кнопка внутри уведомления.

        :param button_data: Информация о кнопке.
        :type button_data: dict
        :param notification_data: Информация об уведомлении.
        :type notification_data: dict
        :return: True, если кнопка находится внутри уведомления, в противном случае - False.
        :rtype: bool
        """

        # Проверка, что кнопка находится внутри уведомления
        button_top_left = button_data["coordinate"]["top_left"]
        button_bottom_right = button_data["coordinate"]["bottom_right"]

        notification_top_left = notification_data["coordinate"]["top_left"]
        notification_bottom_right = notification_data["coordinate"]["bottom_right"]

        return (
                button_top_left["x"] >= notification_top_left["x"]
                and button_top_left["y"] >= notification_top_left["y"]
                and button_bottom_right["x"] <= notification_bottom_right["x"]
                and button_bottom_right["y"] <= notification_bottom_right["y"]
        )

    # endregion

    # region варианты предсказания
    def predict_my_model(self, img_path, show_predict=False):
        """
        Предсказывает объекты на изображении с использованием обученной модели YOLOv8.

        :param img_path: Путь к изображению для предсказания объектов.
        :type img_path: str
        :param show_predict: Флаг для отображения графического представления предсказанных объектов.
        :type show_predict: bool, optional
        :return: Строка JSON с информацией о распознанных объектах.
        :rtype: str
        """

        name_model = "runs/detect/train12/weights/best.pt"
        model = YOLO(name_model)  # Загрузка
        rez_predict = model.predict(img_path)
        rez_json_file, coordinates, categories = self._infer_objects(rez_predict)
        if show_predict and img_path.find(".jpg"):
            self._plot_results(coordinates, categories, img_path)
        return rez_json_file

    def _predict_my_model_in_test_folder(self, show_predict=True):
        # ненужный метод
        images_folder = "C:\\я у мамы программист\\Пробуюсь на работу\\Распознование уведомлений на скриншотах v.2\\Detected_Notification-3\\test\\images"
        image_files = [f for f in os.listdir(images_folder) if f.endswith('.jpg')]

        for image_file in image_files:
            image_path = os.path.join(images_folder, image_file)
            self.predict_my_model(image_path, show_predict)

    def _predict_my_model_in_json(self, show_predict, json_file_path):
        # ненужный метод
        with open(json_file_path, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)

        screenshots = data.get('screenshot', [])
        if not screenshots:
            print("Error: 'screenshot' key not found or empty in the JSON file.")
            return

        for screenshot_path in screenshots:
            screenshot_path = screenshot_path.get('path', '')
            if not screenshot_path:
                print("Error: 'path' key not found in a screenshot entry.")
                continue

            self.predict_my_model(screenshot_path, show_predict)
    # endregion

# json_file_path = "input.json"
# predict_my_model_in_json(True, json_file_path)

# path = "C:\я у мамы программист\Пробуюсь на работу\Распознование уведомлений на скриншотах v.2" \
#        "\Detected_Notification-3\\test\images\\15_jpg.rf.760b27a3ac6533de7fe011666123ecd6.jpg"
# model = CustomYOLOv8Model()
# print(model.predict_my_model(path, True))

# model.train_my_model()
# model.predict_my_model_in_test_folder(True)
