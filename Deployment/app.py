from flask import Flask, request, jsonify, render_template
from inference import generate_poem

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/generate", methods=["POST"])
def generate():
    data = request.json
    prompt = data.get("prompt", "")
    temperature = float(data.get("temperature", 1.0))
    max_tokens = int(data.get("max_tokens", 200))

    output = generate_poem(prompt, max_tokens=max_tokens, temperature=temperature)

    return jsonify({"response": output})

if __name__ == "__main__":
    app.run(debug=True)