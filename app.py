# Import libraries
from flask import Flask, jsonify, request, render_template
import json
import numpy as np
import pickle
import pandas as pd

app = Flask(__name__)
# Load the model

with open("new_model.pkl", "rb") as f:
    model = pickle.load(f)


@app.route('/', methods=["GET", "POST"])
def index():
    pred = ""
    if request.method == "POST":
       # avg_coupon_discount = request.form["avg_coupon_discount"]
       # month_trans_ave = request.form["month_trans_ave"]
       # avg_use_other_discount = request.form["avg_use_other_discount"]
       # cus_Fuel = request.form["cus_Fuel"]
       # cus_Established = request.form["cus_Established"]
       # cus_Meat =  request.form["cus_Meat"]
       # income_bracket_income_11 = request.form["income_bracket_income_11"]
       # brand_type_Local = request.form["brand_type_Local"]
       # income_bracket_income_12 = request.form["income_bracket_income_12"]

       # X = np.array([[float(avg_coupon_discount), float(month_trans_ave), float(avg_use_other_discount),
                 # float(cus_Fuel), float(cus_Established), float(cus_Meat),
                  # float(income_bracket_income_11), float(brand_type_Local), float(income_bracket_income_12)]])
    
        int_features = [x for x in request.form.values()]
        X = np.array(int_features)

        new_X = pd.DataFrame([X], columns = ['avg_coupon_discount',
                                              'month_trans_ave',
                                              'avg_use_other_discount',
                                              'cus_Fuel',
                                              'cus_Established',
                                              'cus_Meat',
                                              'income_bracket_income_11',
                                              'brand_type_Local',
                                              'income_bracket_income_12'])

        pred = model.predict_proba(new_X)[0][1]
    return render_template("index.html", pred = pred)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
