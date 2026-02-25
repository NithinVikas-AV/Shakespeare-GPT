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
    output = generate_poem(prompt)
    return jsonify({"response": output})

if __name__ == "__main__":
    app.run(debug=True)