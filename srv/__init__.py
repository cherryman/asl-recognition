import io
import base64
from flask import Flask, Blueprint, render_template, request
from PIL import Image
from model import model_eval

bp = Blueprint("", __name__)


def create_app():
    app = Flask(__name__)
    app.register_blueprint(bp)
    return app


@bp.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@bp.route("/eval", methods=["POST"])
def eval():
    data = request.get_json()
    image = Image.open(io.BytesIO(base64.b64decode(data['data']))).convert("RGB")
    image.save('./caca.jpg')
    print(model_eval(image))
    return {}
