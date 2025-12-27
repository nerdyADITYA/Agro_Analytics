from django.shortcuts import render,redirect
from Crop.models import User,Crop
from django.http import JsonResponse
from django.conf import settings
from django.db import IntegrityError
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import requests
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import os

# Create your views here.
def signup(request):
    # If user is already logged in, redirect to home
    if 'user_id' in request.session:
        return redirect("home")
    
    # If the request is a POST, process the form data
    if request.method == "POST":
        # Create a new User object with the submitted username, email, and password
        try:
            User.objects.create(
                First_Name = request.POST["first_name"],
                Last_Name = request.POST["last_name"],
                Username = request.POST["username"],
                Email = request.POST["email"],
                Age = request.POST.get("age"),
                Phone_Number = request.POST.get("phone"),
                Password = request.POST["password"]
            )
            # After successful signup, redirect the user to the login page
            return redirect("login")
        except IntegrityError as e:
            msg = str(e).lower()
            if 'email' in msg or 'unique constraint' in msg:
                error_message = 'This email is already registered. Please log in or use a different email.'
            else:
                error_message = 'Unable to create account. Please check your details and try again.'

            # Render the signup page with a friendly error message
            return render(request, "signup.html", {"error": error_message})
    # If the request is GET, render the signup form
    return render(request,"signup.html")

def login(request):
    # If user is already logged in, redirect to home
    if 'user_id' in request.session:
        return redirect("home")
    
    # If the request is a POST, process the login form data
    if request.method == "POST":
        # Retrieve the username and password from the submitted form
        username = request.POST["username"]
        password = request.POST["password"]
        # Check if a user with the given username and password exists
        user = User.objects.filter(Username=username, Password=password).first()
        if user:
            # Store user info in session
            request.session['user_id'] = user.id
            request.session['username'] = user.Username
            # If user exists, redirect to the home page
            return redirect("home")
        else:
            # If user does not exist, render the login page with an error message
            return render(request, "login.html", {"error": "Invalid credentials"})
    # If the request is GET, render the login form
    return render(request, "login.html")

def home(request):
    # Check if user is logged in
    if 'user_id' not in request.session:
        return redirect("login")
    
    # Get user info from session
    user = User.objects.get(id=request.session['user_id'])
    
    # Check if this is a prediction request (has parameters)
    if not any(param in request.GET for param in ['nitrogen', 'phosphorus', 'potassium', 'ph', 'city']):
        # No parameters provided, show the form
        return render(request, 'home.html', {'user': user})

    # Load dataset
    csv_path = os.path.join(settings.BASE_DIR, 'crop', 'Crop_recommendation.csv')
    df = pd.read_csv(csv_path)

    # Features and target
    X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
    y = df['label']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Calculate accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Get input parameters from request
    try:
        nitrogen = request.GET['nitrogen']
        phosphorus = request.GET['phosphorus']
        potassium = request.GET['potassium']
        ph = request.GET['ph']
        city = request.GET['city']
    except KeyError as e:
        return JsonResponse({'error': f'Missing required parameter: {e.args[0]}'}, status=400)

    # Convert to appropriate types
    try:
        nitrogen = float(nitrogen)
        phosphorus = float(phosphorus)
        potassium = float(potassium)
        ph = float(ph)
        city = city.strip()  # Remove any extra spaces
        if not city:
            return JsonResponse({'error': 'City name cannot be empty'}, status=400)
    except ValueError as e:
        return JsonResponse({'error': f'Invalid parameter value. Nitrogen, Phosphorus, Potassium, and pH must be numeric.'}, status=400)

    # OpenWeatherMap API key (replace with your actual key)
    api_key = '415ba6cfae13440c33af57516e83c925'  # Replace with your API key

    # First, get coordinates from city name using Geocoding API
    try:
        geo_url = f'http://api.openweathermap.org/geo/1.0/direct?q={city}&appid={api_key}'
        geo_response = requests.get(geo_url)
        geo_response.raise_for_status()
        geo_data = geo_response.json()
        
        if not geo_data:
            return JsonResponse({'error': f'City "{city}" not found. Please check the spelling and try again.'}, status=400)
        
        # Get coordinates from the first result
        lat = geo_data[0]['lat']
        lon = geo_data[0]['lon']
        country = geo_data[0].get('country', 'Unknown')
        state = geo_data[0].get('state', '')
        
    except Exception as e:
        return JsonResponse({'error': f'Failed to get coordinates for city "{city}": {str(e)}'}, status=400)

    # Fetch weather data using coordinates
    try:
        url = f'http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric'
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        temperature = data['main']['temp']
        humidity = data['main']['humidity']
        
        # Get rainfall - check multiple sources and use reasonable defaults
        rainfall = 0.0
        if 'rain' in data:
            rainfall = data['rain'].get('1h', 0.0)
            if rainfall == 0.0:
                rainfall = data['rain'].get('3h', 0.0) / 3  # Convert 3h to 1h average
        
        # Since dataset minimum rainfall is 20mm, we need to estimate seasonal/monthly rainfall
        # Real-time weather API gives current conditions, but crops need seasonal patterns
        if 'weather' in data:
            main_weather = data['weather'][0]['main'].lower()
            description = data['weather'][0]['description'].lower()
            
            # Estimate based on climate and humidity
            if 'rain' in main_weather or 'drizzle' in main_weather:
                rainfall = max(rainfall, 80.0)  # Active rain = good rainfall region
            elif 'thunderstorm' in main_weather:
                rainfall = max(rainfall, 120.0)  # Thunderstorm = high rainfall region
            elif 'cloud' in main_weather:
                # Cloudy conditions suggest moderate rainfall potential
                if humidity > 80:
                    rainfall = max(rainfall, 90.0)  # High humidity + clouds = good rainfall
                elif humidity > 60:
                    rainfall = max(rainfall, 60.0)  # Moderate humidity + clouds
                else:
                    rainfall = max(rainfall, 40.0)  # Low humidity + clouds
            elif 'clear' in main_weather:
                # Clear weather - estimate based on humidity and temperature
                if humidity > 80:
                    rainfall = max(rainfall, 70.0)  # High humidity = potential for rain
                elif humidity > 60:
                    rainfall = max(rainfall, 50.0)  # Moderate humidity
                else:
                    rainfall = max(rainfall, 30.0)  # Low humidity = dry region
            else:
                # Default estimation based on humidity
                if humidity > 80:
                    rainfall = max(rainfall, 80.0)
                elif humidity > 60:
                    rainfall = max(rainfall, 60.0)
                else:
                    rainfall = max(rainfall, 40.0)
        else:
            # Fallback: estimate based on humidity alone
            if humidity > 80:
                rainfall = 90.0
            elif humidity > 60:
                rainfall = 70.0
            else:
                rainfall = 45.0
        
        weather_source = "OpenWeatherMap API"
    except Exception as e:
        return JsonResponse({'error': f'Failed to fetch weather data for "{city}": {str(e)}. Please check your API key and city name.'}, status=400)

    # Prepare features for prediction
    feature_names = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    features = pd.DataFrame([[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]], columns=feature_names)
    probabilities = model.predict_proba(features)[0]
    crop_names = model.classes_

    # Sort crops by probability
    crop_probs = list(zip(crop_names, probabilities))
    crop_probs.sort(key=lambda x: x[1], reverse=True)

    # Get top 5 crops
    top5_crops = crop_probs[:5]
    best_crop = crop_probs[0][0]

    # Create bar graph for top 5 crops
    crops, probs = zip(*top5_crops)
    plt.figure(figsize=(12, 6))
    bars = plt.bar(crops, [p*100 for p in probs], color='skyblue')
    plt.title('Top 5 Crop Recommendations', fontsize=16)
    plt.xlabel('Crops', fontsize=12)
    plt.ylabel('Probability (%)', fontsize=12)
    plt.xticks(rotation=45)
    
    for bar, prob in zip(bars, probs):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{prob * 100:.1f}%', ha='center', va='bottom')

    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plot_data = base64.b64encode(buffer.read()).decode('utf-8')
    buffer.close()
    plt.close()

    # Save crop data to database
    Crop.objects.create(
        nitrogen=nitrogen,
        phosphorus=phosphorus,
        potassium=potassium,
        ph=ph,
        temprature=temperature,
        humidity=humidity,
        rainfall=rainfall,
    )

    # Prepare context data
    context = {
        'best_crop': best_crop,
        'confidence': crop_probs[0][1] * 100,
        'accuracy': accuracy * 100,
        'top5_crops': [(crop, prob * 100) for crop, prob in top5_crops],
        'plot_data': plot_data,
        'input_data': {
            'nitrogen': nitrogen,
            'phosphorus': phosphorus,
            'potassium': potassium,
            'ph': ph,
            'temperature': temperature,
            'humidity': humidity,
            'rainfall': rainfall,
            'city': city,
            'country': country,
            'state': state,
            'latitude': lat,
            'longitude': lon,
            'weather_source': weather_source
        }
    }

    # Add user to context
    context['user'] = user
    
    # Return JSON response if requested
    if request.headers.get('Content-Type') == 'application/json' or request.GET.get('format') == 'json':
        return JsonResponse(context)

    return render(request, 'home.html', context)

def profile(request):
    # Check if user is logged in
    if 'user_id' not in request.session:
        return redirect("login")
    
    user = User.objects.get(id=request.session['user_id'])
    
    # Handle profile update
    if request.method == "POST":
        try:
            user.First_Name = request.POST["first_name"]
            user.Last_Name = request.POST["last_name"]
            user.Email = request.POST["email"]
            user.Age = request.POST["age"]
            user.Phone_Number = request.POST["phone"]
            
            # Only update password if provided
            new_password = request.POST.get("password")
            if new_password and new_password.strip():
                user.Password = new_password
            
            user.save()
            
            return render(request, 'profile.html', {
                'user': user, 
                'success': 'Profile updated successfully!'
            })
        except Exception as e:
            return render(request, 'profile.html', {
                'user': user, 
                'error': f'Error updating profile: {str(e)}'
            })
    
    return render(request, 'profile.html', {'user': user})

def logout(request):
    # Clear the session
    if 'user_id' in request.session:
        del request.session['user_id']
    if 'username' in request.session:
        del request.session['username']
    
    # Redirect to login page
    return redirect("login")
