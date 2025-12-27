from django.shortcuts import render, redirect
from Crop.models import User, Crop
from django.http import JsonResponse
from django.conf import settings
from django.db import IntegrityError
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import requests
import matplotlib
matplotlib.use("Agg")  # Non-GUI backend for production
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import os

# =========================================================
# LOAD DATASET & TRAIN MODEL ONCE (IMPORTANT FOR PRODUCTION)
# =========================================================

CSV_PATH = os.path.join(settings.BASE_DIR, "Crop", "Crop_recommendation.csv")

df = pd.read_csv(CSV_PATH)

X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = df['label']

# Train-test split ONLY for accuracy calculation
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

MODEL = RandomForestClassifier(n_estimators=100, random_state=42)
MODEL.fit(X_train, y_train)

y_pred = MODEL.predict(X_test)
MODEL_ACCURACY = accuracy_score(y_test, y_pred)

# =========================================================
# AUTH VIEWS
# =========================================================

def signup(request):
    if 'user_id' in request.session:
        return redirect("home")

    if request.method == "POST":
        try:
            User.objects.create(
                First_Name=request.POST["first_name"],
                Last_Name=request.POST["last_name"],
                Username=request.POST["username"],
                Email=request.POST["email"],
                Age=request.POST.get("age"),
                Phone_Number=request.POST.get("phone"),
                Password=request.POST["password"]
            )
            return redirect("login")
        except IntegrityError as e:
            msg = str(e).lower()
            error_message = (
                "This email is already registered."
                if "email" in msg else
                "Unable to create account."
            )
            return render(request, "signup.html", {"error": error_message})

    return render(request, "signup.html")


def login(request):
    if 'user_id' in request.session:
        return redirect("home")

    if request.method == "POST":
        user = User.objects.filter(
            Username=request.POST["username"],
            Password=request.POST["password"]
        ).first()

        if user:
            request.session["user_id"] = user.id
            request.session["username"] = user.Username
            return redirect("home")

        return render(request, "login.html", {"error": "Invalid credentials"})

    return render(request, "login.html")


def logout(request):
    request.session.flush()
    return redirect("login")

# =========================================================
# HOME / PREDICTION VIEW
# =========================================================

def home(request):
    if 'user_id' not in request.session:
        return redirect("login")

    user = User.objects.get(id=request.session['user_id'])

    # If form not submitted
    if not request.GET:
        return render(request, "home.html", {"user": user})

    # -----------------------------------------------------
    # INPUT PARSING
    # -----------------------------------------------------

    try:
        nitrogen = float(request.GET["nitrogen"])
        phosphorus = float(request.GET["phosphorus"])
        potassium = float(request.GET["potassium"])
        ph = float(request.GET["ph"])
        city = request.GET["city"].strip()
    except Exception:
        return JsonResponse({"error": "Invalid input values"}, status=400)

    # -----------------------------------------------------
    # WEATHER API
    # -----------------------------------------------------

    OPENWEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY")
    if not OPENWEATHER_API_KEY:
        return JsonResponse({"error": "Weather API key missing"}, status=500)

    # Geocoding
    geo_url = f"https://api.openweathermap.org/geo/1.0/direct?q={city}&appid={OPENWEATHER_API_KEY}"
    geo_data = requests.get(geo_url).json()

    if not geo_data:
        return JsonResponse({"error": "City not found"}, status=400)

    lat = geo_data[0]["lat"]
    lon = geo_data[0]["lon"]
    country = geo_data[0].get("country", "")
    state = geo_data[0].get("state", "")

    # Weather
    weather_url = (
        f"https://api.openweathermap.org/data/2.5/weather"
        f"?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric"
    )

    data = requests.get(weather_url).json()

    temperature = data["main"]["temp"]
    humidity = data["main"]["humidity"]

    rainfall = 50.0
    if "rain" in data:
        rainfall = data["rain"].get("1h", 50.0)

    # -----------------------------------------------------
    # MODEL PREDICTION
    # -----------------------------------------------------

    features = pd.DataFrame(
        [[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]],
        columns=X.columns
    )

    probabilities = MODEL.predict_proba(features)[0]
    crops = MODEL.classes_

    crop_probs = sorted(
        zip(crops, probabilities),
        key=lambda x: x[1],
        reverse=True
    )

    top5 = crop_probs[:5]
    best_crop = top5[0][0]

    # -----------------------------------------------------
    # PLOT GENERATION
    # -----------------------------------------------------

    crop_names, probs = zip(*top5)

    plt.figure(figsize=(10, 5))
    plt.bar(crop_names, [p * 100 for p in probs], color="green")
    plt.title("Top 5 Crop Recommendations")
    plt.ylabel("Probability (%)")
    plt.xticks(rotation=30)

    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    plot_data = base64.b64encode(buffer.read()).decode()
    buffer.close()
    plt.close()

    # -----------------------------------------------------
    # SAVE TO DATABASE
    # -----------------------------------------------------

    Crop.objects.create(
        nitrogen=nitrogen,
        phosphorus=phosphorus,
        potassium=potassium,
        ph=ph,
        temprature=temperature,
        humidity=humidity,
        rainfall=rainfall,
    )

    # -----------------------------------------------------
    # RESPONSE
    # -----------------------------------------------------

    context = {
    "user": user,
    "best_crop": best_crop,
    "confidence": top5[0][1] * 100,
    "accuracy": MODEL_ACCURACY * 100,
    "top5_crops": [(c, p * 100) for c, p in top5],
    "plot_data": plot_data,
    "input_data": {
        "city": city,
        "country": country,
        "state": state,
        "latitude": lat,
        "longitude": lon,
        "nitrogen": nitrogen,
        "phosphorus": phosphorus,
        "potassium": potassium,
        "ph": ph,
        "temperature": temperature,
        "humidity": humidity,
        "rainfall": rainfall,
        "weather_source": "OpenWeatherMap API"
    }
}


    return render(request, "home.html", context)

# =========================================================
# PROFILE
# =========================================================

def profile(request):
    if 'user_id' not in request.session:
        return redirect("login")

    user = User.objects.get(id=request.session["user_id"])

    if request.method == "POST":
        user.First_Name = request.POST["first_name"]
        user.Last_Name = request.POST["last_name"]
        user.Email = request.POST["email"]
        user.Age = request.POST["age"]
        user.Phone_Number = request.POST["phone"]

        if request.POST.get("password"):
            user.Password = request.POST["password"]

        user.save()
        return render(request, "profile.html", {"user": user, "success": "Profile updated"})

    return render(request, "profile.html", {"user": user})
