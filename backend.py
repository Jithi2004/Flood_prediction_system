import requests
import pandas as pd
import pickle
from flask import Flask, request, jsonify, render_template
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score

app = Flask(__name__)

# Function to fetch real-time weather data from OpenWeather API
def fetch_weather_data(city):
    """Fetches weather data from OpenWeather API."""
    Api_key = '6e6f9659fef62e5c5d1103979100d281'
    base_url = 'http://api.openweathermap.org/data/2.5/weather'
    request_url = f"{base_url}?appid={Api_key}&q={city}"
    
    try:
        response = requests.get(request_url)
        response.raise_for_status()
        data = response.json()
        weather = {
            "Temperature": round(data['main']['temp'] - 273.15, 2),
            "Humidity": data['main']['humidity'],
            "WindSpeed": round(data['wind']['speed'] * 3.6, 2),
            "Pressure": data['main']['pressure'],
            "CloudCover": data['clouds']['all'],
            "Rainfall": data.get('rain', {}).get('1h', 0),
        }
        return weather
    except requests.exceptions.RequestException as e:
        print(f"Error fetching weather data: {e}")
        return None

# Function to train the flood prediction model
def train_flood_model():
    """Trains a Gradient Boosting model for flood risk prediction."""
    df = pd.read_csv('flood.csv')
    X = df.drop(columns=["FloodProbability"])
    y = df["FloodProbability"]

    # Calculate the mean of static features to use later
    static_feature_means = X.iloc[:, 5:].mean().tolist()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    gb_model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=42)
    gb_model.fit(X_train, y_train)

    y_pred = gb_model.predict(X_test)
    gb_accuracy = r2_score(y_test, y_pred)
    print(f"Gradient Boosting Model R² Score: {gb_accuracy:.2f}")

    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open("static_means.pkl", "wb") as f:
        pickle.dump(static_feature_means, f)
    with open("gradient_boosting.pkl", "wb") as f:
        pickle.dump(gb_model, f)

# Function to predict flood probability
def predict_flood_risk(weather_data):
    """Predicts flood probability based on weather data."""
    try:
        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        with open("gradient_boosting.pkl", "rb") as f:
            model = pickle.load(f)
        with open("static_means.pkl", "rb") as f:
            static_means = pickle.load(f)

        input_features = [
            weather_data["Temperature"],
            weather_data["Humidity"],
            weather_data["Rainfall"],
            weather_data["WindSpeed"],
            weather_data["Pressure"],
        ] + static_means

        input_scaled = scaler.transform([input_features])
        prediction = model.predict(input_scaled)
        return prediction[0]
    except Exception as e:
        print(f"Error in prediction: {e}")
        return None

# Flask Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handles prediction requests from the frontend."""
    city = request.form['city']
    weather_data = fetch_weather_data(city)

    if weather_data:
        flood_risk = predict_flood_risk(weather_data)
        if flood_risk is not None:
            return jsonify({"flood_risk": f"{flood_risk * 100:.2f}%"})
        else:
            return jsonify({"error": "Prediction failed due to incorrect input format."})
    else:
        return jsonify({"error": "Failed to fetch weather data"})

# Run the Flask app
if __name__ == "__main__":
    train_flood_model()
    app.run(debug=True)
