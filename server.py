import io
import base64
from flask import Flask, Blueprint, render_template, request
from PIL import Image
from model import new_net, model_eval_prep, model_eval

net = model_eval_prep(new_net())
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
    global net
    data = request.get_json()
    image = Image.open(io.BytesIO(base64.b64decode(data["data"]))).convert("RGB")
    return {"data": model_eval(net, image)}
