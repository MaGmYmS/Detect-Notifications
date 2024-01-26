import base64
import unittest

import requests


class TestMyAPI(unittest.TestCase):
    def test_predict_endpoint(self):
        # Подготовка изображения в формате base64
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


if __name__ == "__main__":
    unittest.main()
