from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load models
classifier_model_path = r"C:\Users\rajka\OneDrive\Desktop\All Folders\My_projects\Predictive_Maintenance_Project_IBM\predictive_maintain_classifier.pkl"
regressor_model_path = r"C:\Users\rajka\OneDrive\Desktop\All Folders\My_projects\Predictive_Maintenance_Project_IBM\predictive_maintain_regressor.pkl"

with open(classifier_model_path, "rb") as f:
    classifier_model = pickle.load(f)

with open(regressor_model_path, "rb") as f:
    regressor_model = pickle.load(f)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Collect form data
        product_id = request.form.get("product_id", "M14860")
        type_val = request.form.get("type", "L")
        air_temp = float(request.form.get("air_temp", 300.7))
        process_temp = float(request.form.get("process_temp", 310.6))
        rot_speed = float(request.form.get("rot_speed", 1452))
        torque = float(request.form.get("torque", 40.2))
        tool_wear = float(request.form.get("tool_wear", 0))

        # Checkbox values
        machine_failure = 1 if request.form.get("machine_failure") == "on" else 0
        TWF = 1 if request.form.get("TWF") == "on" else 0
        HDF = 1 if request.form.get("HDF") == "on" else 0
        PWF = 1 if request.form.get("PWF") == "on" else 0
        OSF = 1 if request.form.get("OSF") == "on" else 0
        RNF = 1 if request.form.get("RNF") == "on" else 0

        # Prepare DataFrames with correct column names
        classifier_columns = ['Product ID', 'Type', 'Air temperature [K]', 'Process temperature [K]',
                              'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]',
                              'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
        regressor_columns = ['Product ID', 'Type', 'Air temperature [K]', 'Process temperature [K]',
                             'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]',
                             'Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']

        classifier_df = pd.DataFrame([[product_id, type_val, air_temp, process_temp,
                                       rot_speed, torque, tool_wear, TWF, HDF, PWF, OSF, RNF]],
                                     columns=classifier_columns)

        regressor_df = pd.DataFrame([[product_id, type_val, air_temp, process_temp,
                                      rot_speed, torque, tool_wear, machine_failure,
                                      TWF, HDF, PWF, OSF, RNF]],
                                    columns=regressor_columns)

        # Predict
        maintenance_pred = classifier_model.predict(classifier_df)[0]
        maintenance_prob = classifier_model.predict_proba(classifier_df)[0][1] if hasattr(classifier_model, "predict_proba") else None
        rul_pred = regressor_model.predict(regressor_df)[0]

        # Pass results to HTML
        result = {
            "maintenance_label": "Failure Likely" if maintenance_pred == 1 else "No Failure",
            "maintenance_prob": round(maintenance_prob, 4) if maintenance_prob is not None else None,
            "rul": round(rul_pred, 2)
        }

        return render_template("index.html", result=result)

    except Exception as e:
        return render_template("index.html", error=str(e))


if __name__ == "__main__":
    app.run(debug=True)
