from flask import Flask, request, render_template
import joblib

app = Flask(__name__)
model = joblib.load("naive_bayes_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        message = request.form["message"]
        prediction = model.predict([message])[0]
        return render_template("index.html", prediction=prediction, message=message)

if __name__ == "__main__":
    app.run(debug=True)
