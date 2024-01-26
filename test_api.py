import base64
import unittest

import requests
from flasgger import Swagger
from flask import Flask

app = Flask(__name__)


class TestMyAPI(unittest.TestCase):
    def test_successful_predict_endpoint(self):
        """
         Тест успешного запроса к /predict.

         ---
         tags:
           - Тестирование
         responses:
           200:
             description: Успешный запрос к /predict
         x-testInfo:
           description: Это тестовая информация для успешного теста
           value: "Тест успешно пройден"
         """
        path_to_img = "C:\я у мамы программист\Пробуюсь на работу\Распознование уведомлений на скриншотах v.2" \
                      "\Detected_Notification-3\\test\images\\15_jpg.rf.760b27a3ac6533de7fe011666123ecd6.jpg"
        with open(path_to_img, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

        # Формирование JSON-запроса
        data = {"screenshot": encoded_image}

        # Отправка POST-запроса на API
        response = requests.post("http://127.0.0.1:5000/predict", json=data)

        # Проверка, что статус код ответа - 200 (успех)
        self.assertEqual(response.status_code, 200)

        # Проверка структуры JSON-ответа
        result_json = response.json()
        self.assertIn("detected_notifications", result_json)
        # Дополнительные проверки успешного сценария, если необходимо

    def test_unsuccessful_predict_endpoint(self):
        """
        Тест неуспешного запроса к /predict.

        ---
        tags:
          - Тестирование
        responses:
          400:
            description: Некорректный запрос к /predict
        x-testInfo:
          description: Это тестовая информация для неуспешного теста
          value: "Тест завершился неудачей"
        """
        # Формирование JSON-запроса
        data = {"screenshot": "incorrect_string"}

        # Отправка POST-запроса на API
        response = requests.post("http://127.0.0.1:5000/predict", json=data)

        result_json = response.json()
        self.assertNotIn("detected_notifications", result_json)


if __name__ == '__main__':
    Swagger(app)  # Подключение Swagger для документации
    unittest.main()
