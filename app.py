from flask import Flask, render_template, request
import numpy as np
from joblib import load

app = Flask(__name__)


@app.route("/", methods=['GET', 'POST'])
def home():
    request_method_type = request.method
    if request_method_type == 'GET':
        return render_template("index.html", output="Welcome to Iris Variety Prediction")
    else:

        model = load('decision_tree_model.joblib')

        input = request.form['input_data']

        input_np_array = string_to_array(input)

        pred = model.predict(input_np_array)[0]

        pred_as_str = str(pred)

        return render_template("index.html", output=pred_as_str)


def string_to_array(input):

    def is_float(x):
        try:
            float(x)
            return True
        except:
            return False

    array = np.array([float(x) for x in input.split(",") if is_float(x)])
    return array.reshape(1, len(array))


if __name__ == "__main__":
    app.run(debug=True, port=8000)