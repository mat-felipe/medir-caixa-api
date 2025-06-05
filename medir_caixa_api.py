# medir_caixa_api.py
from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64

app = Flask(__name__)

# Largura real do quadrado marcador em centímetros
MARKER_WIDTH_CM = 5.0

@app.route('/processar-imagem', methods=['POST'])
def processar_imagem():
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'erro': 'Imagem não recebida'}), 400

        image_data = base64.b64decode(data['image'])
        npimg = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({'erro': 'Falha ao decodificar imagem'}), 400

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 50, 150)

        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return jsonify({'erro': 'Nenhum contorno encontrado'}), 400

        # Detecta marcador quadrado de referência (5x5cm)
        marker = None
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / float(h)
            area = cv2.contourArea(cnt)
            if 0.9 < aspect_ratio < 1.1 and 500 < area < 5000:
                marker = (w, h)
                break

        if marker is None:
            return jsonify({'erro': 'Marcador de referência não encontrado'}), 400

        pixels_per_cm = marker[0] / MARKER_WIDTH_CM

        # Detecta a maior caixa (assume que seja o maior contorno)
        box_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(box_contour)

        comprimento = round(w / pixels_per_cm, 1)
        largura = round(h / pixels_per_cm, 1)
        altura = round(min(comprimento, largura) / 2, 1)  # heurística

        return jsonify({
            'length': comprimento,
            'width': largura,
            'height': altura
        })

    except Exception as e:
        return jsonify({'erro': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
