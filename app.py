from flask import Flask, request, jsonify
import sklearn
import numpy as np
import pickle

model = pickle.load(open('model.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def index():
    return "Hello World"

@app.route('/predict', methods=['POST'])
def predict():

   fulfill_via = request.form.get('fulfill_via')
   shipment_mode = request.form.get('shipment_mode')
   product_group = request.form.get('product_group')
   sub_classification = request.form.get('sub_classification')
   line_item_quantity = request.form.get('line_item_quantity')
   pack_price = request.form.get('pack_price')
   unit_price = request.form.get('unit_price')
   weight = request.form.get('weight')
   dosage_numerical = request.form.get('dosage_numerical')

   input_query = np.array([[fulfill_via, shipment_mode,
                             product_group, sub_classification, line_item_quantity, pack_price,
                             unit_price, weight, dosage_numerical]])

   result = model.predict(input_query)[0]
   return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
