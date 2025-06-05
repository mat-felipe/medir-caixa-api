# medir_caixa_api.py
from flask import Flask, request, jsonify
import cv2
import numpy as np
import tempfile
import os

app = Flask(__name__)

# Configurável: tamanho real do marcador em centímetros
MARKER_WIDTH_CM = 5.0

@app.route('/processar-imagem', methods=['POST'])
def processar_imagem():
    try:
        # Recebe a imagem
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'erro': 'Imagem não recebida'}), 400

        # Decodifica base64
        import base64
        image_data = base64.b64decode(data['image'])
        npimg = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        # Converte para escala de cinza e aplica blur
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 50, 150)

        # Encontra contornos
        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return jsonify({'erro': 'Nenhum contorno encontrado'}), 400

        # Ordena por área e assume que o maior contorno é a caixa
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        box_contour = contours[0]
        x, y, w, h = cv2.boundingRect(box_contour)

        # Detecta o marcador como o menor contorno retangular visível (opcional: ArUco, QR, etc.)
        marker_contour = min(contours, key=cv2.contourArea)
        _, _, mw, _ = cv2.boundingRect(marker_contour)
        if mw == 0:
            return jsonify({'erro': 'Erro na medição do marcador'}), 400

        # Proporção pixels/cm
        pixels_per_cm = mw / MARKER_WIDTH_CM

        comprimento = round(w / pixels_per_cm, 1)
        largura = round(h / pixels_per_cm, 1)
        altura = round(min(comprimento, largura) / 2, 1)  # Heurística: altura estimada

        return jsonify({
            'length': comprimento,
            'width': largura,
            'height': altura
        })

    except Exception as e:
        return jsonify({'erro': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
