from flask import Flask, render_template, request
import pickle
import os

app = Flask(__name__)

# Paths to your models
classifier_path = r"C:\Users\rajka\OneDrive\Desktop\All Folders\My_projects\Predictive_Maintenance_Project_IBM\predictive_maintain_classifier.pkl"
regressor_path = r"C:\Users\rajka\OneDrive\Desktop\All Folders\My_projects\Predictive_Maintenance_Project_IBM\predictive_maintain_regressor.pkl"

# Load models
with open(classifier_path, "rb") as f:
    classifier_model = pickle.load(f)

with open(regressor_path, "rb") as f:
    regressor_model = pickle.load(f)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        feature1 = float(request.form["feature1"])
        feature2 = float(request.form["feature2"])
        feature3 = float(request.form["feature3"])
        feature4 = float(request.form["feature4"])

        features = [[feature1, feature2, feature3, feature4]]

        # Classifier prediction
        classification = classifier_model.predict(features)[0]

        # Regressor prediction
        regression = regressor_model.predict(features)[0]

        return render_template(
            "index.html",
            prediction_text=f"Classification: {classification}, Regression: {regression}",
        )
    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")


if __name__ == "__main__":
    app.run(debug=True)
