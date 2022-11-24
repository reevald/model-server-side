from flask import Flask, jsonify, request
from flask_cors import CORS
from io import BytesIO
from PIL import Image
import tensorflow as tf
import logging
import base64
import binascii

app = Flask(__name__)
CORS(app, support_credentials=True)
logging.basicConfig(level=logging.INFO)

label = ["Paper (Kertas)", "Rock (Batu)", "Scissor (Gunting)"]
logging.info("[Start] Load model")
model = tf.keras.models.load_model(r"model/model-v2.hdf5")
logging.info("[Finish] Load model")


@app.get("/")
def home():
    return "Hello World :) "


@app.get("/status-model")
def status_model():
    if model is None:
        return "Model not yet loaded"
    else:
        stringlist = []
        model.summary(print_fn=lambda x: stringlist.append(x))
        short_model_summary = "\n".join(stringlist)
        return short_model_summary


@app.post("/classifier")
def classifier():
    # 1) Cek dulu apakah model sudah dimuat atau belum
    if model is None:
        raise RuntimeError("Model not yet loaded")

    # 2) Mengambil data yang dikirim
    data = dict(request.form)

    # 3) Cek apakah data base64 valid
    try:
        img_bytes = base64.b64decode(data["imgBase64"])
    except binascii.Error:
        raise RuntimeError("Invalid base64 image data")

    # 4) Cek apakah dapat dikonversi ke data image
    try:
        img = Image.open(BytesIO(img_bytes))
    except Exception:
        raise RuntimeError("Failed to load image data")

    # 5) Menyesuaikan ukuran tensor dengan ukuran input pada model
    img = tf.image.resize(img, size=(150, 150))

    # 6) Lakukan normalisasi ukuran pixel dari (0, 255) -> (0, 1)
    img = tf.math.divide(img, 255)

    # 7) Sesuaikan dimensi tensor dengan dimensi input pada model
    # Contoh: dari [x] menjadi [[x]], x = tensor
    img = tf.expand_dims(img, axis=0)

    # Melakukan prediksi dan mengirim respon
    prediction = model.predict(img)[0]
    detail_pred = []
    for i in range(len(prediction)):
        detail_pred.append(
            {"class": label[i], "confidence": str(prediction[i])}
        )
    return jsonify(detail_pred)


if __name__ == "__main__":

    app.run()
