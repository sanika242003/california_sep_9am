from flask import Flask, request, render_template
import joblib

# Load model
obj = joblib.load('california.joblib')

model = obj['Model']
columns = obj['Columns']

print("Columns:", columns)

app = Flask(__name__)


@app.route('/')
def Main():
    return render_template('index.html', columns=columns)


@app.route('/predict', methods=['POST'])
def predict():

    Input = []

    for column in columns:
        value = float(request.form[column])
        Input.append(value)

    print("Input:", Input)

    prediction = model.predict([Input])

    print("Prediction:", prediction)

    result = prediction[0]

    return render_template(
        'index.html',
        columns=columns,
        prediction=result
    )


if __name__ == "__main__":
    app.run(debug=True)