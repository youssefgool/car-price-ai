from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained model
model = joblib.load("car_price_model.pkl")

# Brand -> Models
car_data = {
    "Toyota": ["Corolla", "Yaris"],
    "Hyundai": ["Elantra", "Accent"],
    "Kia": ["Cerato", "Sportage"],
    "Nissan": ["Sunny", "Qashqai"],
    "BMW": ["316", "320"],
    "Mercedes": ["C180", "E200"],
    "Chevrolet": ["Optra", "Cruze"],
    "Honda": ["Civic", "City"]
}

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        brand = request.form["brand"]
        model_name = request.form["model"]
        year = int(request.form["year"])
        km_driven = int(request.form["km_driven"])

        input_data = pd.DataFrame([{
            "brand": brand,
            "model": model_name,
            "year": year,
            "km_driven": km_driven
        }])

        predicted_price = model.predict(input_data)[0]
        prediction = f"{predicted_price:,.0f} EGP"

    return render_template(
        "index.html",
        car_data=car_data,
        prediction=prediction
    )

if __name__ == "__main__":
    app.run(debug=True)