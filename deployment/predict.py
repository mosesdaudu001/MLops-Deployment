data = {
    "age": 30,	
    "job": "unemployed",	
    "marital": "married",	
    "education": "primary",	
    "default": "no",	
    "balance": 1787,	
    "housing": "no",	
    "loan": "no",	
    "contact": "cellular",	
    "day": 19,
    "month": "oct",
    "duration": 79,
    "campaign": 1,
    "pdays": -1,
    "previous": 0,
    "poutcome": "unknown"
}
categorical_columns = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "poutcome"
 ]

numerical_columns = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']


## Start
import pickle
from flask import Flask, request, jsonify
model_file = 'model_C=1.0.bin'

app = Flask('pred')

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # dicts = df[categorical_columns + numerical_columns].to_dict(orient='records')

    X = dv.transform(data)
    y_pred = model.predict(X)
    if y_pred[0] == 0:
        ans = "do not give loan"
    else:
        ans = 'You may give loan'

    result = {
        'Should we issue loan': int(y_pred[0]),
        'ans': ans
    }

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)


# waitress-serve --listen=0.0.0.0:9696 predict:app 