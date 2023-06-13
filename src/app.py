from flask import Flask, send_file
from io import BytesIO
import canvas

app = Flask(__name__)
renderer = canvas.renderer()

@app.route("/")
def index():
    return send_file('static/index.html')

@app.route("/stream")
def stream():
    img = next(renderer)
    byte_io = BytesIO()
    img.save(byte_io, 'BMP')
    byte_io.seek(0)
    return send_file(byte_io, mimetype='image/bmp')

if __name__ == "__main__":
    app.run(port=5000, debug=True, host='0.0.0.0')
