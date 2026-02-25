from flask import Flask, request, jsonify, render_template
from inference import generate_poem
import json

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/models", methods=["GET"])
def get_models():
    with open("models/models.json", "r") as f:
        models = json.load(f)
    return jsonify(models)


@app.route("/generate", methods=["POST"])
def generate():

    data = request.json

    if not data:
        return jsonify({"error": "Invalid request"}), 400

    model_name = data.get("model")
    prompt = data.get("prompt", "").strip()
    temperature = float(data.get("temperature", 1.0))
    max_tokens = int(data.get("max_tokens", 200))

    if not model_name:
        return jsonify({"error": "Model not specified"}), 400

    try:
        output = generate_poem(
            model_name,
            prompt,
            max_tokens=max_tokens,
            temperature=temperature
        )
        return jsonify({"response": output})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)