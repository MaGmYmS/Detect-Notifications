import base64
from io import BytesIO

from CustomYOLOv8Model import CustomYOLOv8Model  # Замените на ваш модуль
from PIL import Image
from flask import Flask, request, jsonify
from flasgger import Swagger

app = Flask(__name__)
Swagger(app)

model = CustomYOLOv8Model()


@app.route('/predict', methods=['POST'])
def predict():
    """
    Метод предсказания для вашего API.

    ---
    parameters:
      - name: screenshot
        in: body
        type: string
        required: true
        description: Кодированное в формате Base64 изображение снимка экрана
    responses:
      200:
        description: Результат предсказания в формате JSON
    """
    try:
        # Получите данные из запроса
        data = request.get_json()
        screenshot_base64 = data['screenshot']

        # Декодируйте изображение из base64
        image_data = base64.b64decode(screenshot_base64)
        image = Image.open(BytesIO(image_data))

        # Выполните предсказание с использованием вашей модели
        result = model.predict_my_model(image)  # Замените на ваш метод предсказания
        # Верните результат в формате JSON
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(port=5000)
