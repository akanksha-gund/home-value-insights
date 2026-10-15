from flask import Flask, render_template_string, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained linear regression model
with open('linear.pkl', 'rb') as f:
    model = pickle.load(f)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>House Price Prediction</title>
    <style>
        :root {
            --primary-color: #4f46e5;
            --primary-hover: #4338ca;
            --bg-color: #f3f4f6;
            --card-bg: #ffffff;
            --text-color: #1f2937;
            --border-color: #d1d5db;
        }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-color);
            margin: 0;
            padding: 20px;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
        }
        .container {
            background: var(--card-bg);
            padding: 40px;
            border-radius: 12px;
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
            max-width: 650px;
            width: 100%;
        }
        h1 {
            text-align: center;
            color: var(--primary-color);
            margin-bottom: 8px;
            font-size: 26px;
        }
        p.subtitle {
            text-align: center;
            color: #6b7280;
            margin-bottom: 30px;
            font-size: 14px;
        }
        .form-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }
        .form-group {
            margin-bottom: 10px;
        }
        .form-group.full-width {
            grid-column: span 2;
        }
        label {
            display: block;
            margin-bottom: 6px;
            font-weight: 600;
            font-size: 13px;
            color: #374151;
        }
        input {
            width: 100%;
            padding: 10px 12px;
            border: 1px solid var(--border-color);
            border-radius: 6px;
            font-size: 14px;
            box-sizing: border-box;
            transition: border-color 0.2s, box-shadow 0.2s;
        }
        input:focus {
            outline: none;
            border-color: var(--primary-color);
            box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.1);
        }
        button {
            width: 100%;
            background-color: var(--primary-color);
            color: white;
            border: none;
            padding: 12px;
            font-size: 16px;
            font-weight: 600;
            border-radius: 6px;
            cursor: pointer;
            transition: background-color 0.2s;
            margin-top: 15px;
        }
        button:hover {
            background-color: var(--primary-hover);
        }
        .result-card {
            margin-top: 25px;
            padding: 20px;
            background-color: #e0e7ff;
            border-left: 5px solid var(--primary-color);
            border-radius: 6px;
            text-align: center;
        }
        .result-card h3 {
            margin: 0;
            color: #312e81;
            font-size: 16px;
            font-weight: 500;
        }
        .result-card .price {
            font-size: 30px;
            font-weight: bold;
            color: var(--primary-color);
            margin-top: 8px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>House Price Predictor</h1>
        <p class="subtitle">Enter property details below to estimate market value</p>
        
        <form method="POST">
            <div class="form-grid">
                <div class="form-group">
                    <label for="Square_Footage">Square Footage</label>
                    <input type="number" step="any" id="Square_Footage" name="Square_Footage" placeholder="e.g., 2000" required>
                </div>
                <div class="form-group">
                    <label for="Num_Bedrooms">Number of Bedrooms</label>
                    <input type="number" id="Num_Bedrooms" name="Num_Bedrooms" placeholder="e.g., 3" required>
                </div>
                <div class="form-group">
                    <label for="Num_Bathrooms">Number of Bathrooms</label>
                    <input type="number" step="any" id="Num_Bathrooms" name="Num_Bathrooms" placeholder="e.g., 2" required>
                </div>
                <div class="form-group">
                    <label for="Year_Built">Year Built</label>
                    <input type="number" id="Year_Built" name="Year_Built" placeholder="e.g., 2010" required>
                </div>
                <div class="form-group">
                    <label for="Lot_Size">Lot Size</label>
                    <input type="number" step="any" id="Lot_Size" name="Lot_Size" placeholder="e.g., 5000" required>
                </div>
                <div class="form-group">
                    <label for="Garage_Size">Garage Size (Cars)</label>
                    <input type="number" step="any" id="Garage_Size" name="Garage_Size" placeholder="e.g., 2" required>
                </div>
                <div class="form-group full-width">
                    <label for="Neighborhood_Quality">Neighborhood Quality (Scale 1-10)</label>
                    <input type="number" step="any" id="Neighborhood_Quality" name="Neighborhood_Quality" placeholder="e.g., 8" required>
                </div>
            </div>
            <button type="submit">Predict Price</button>
        </form>

        {% if prediction is not none %}
        <div class="result-card">
            <h3>Estimated House Price</h3>
            <div class="price">${{ "{:,.2f}".format(prediction) }}</div>
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    if request.method == 'POST':
        try:
            # Extract features in the precise training order
            square_footage = float(request.form['Square_Footage'])
            num_bedrooms = float(request.form['Num_Bedrooms'])
            num_bathrooms = float(request.form['Num_Bathrooms'])
            year_built = float(request.form['Year_Built'])
            lot_size = float(request.form['Lot_Size'])
            garage_size = float(request.form['Garage_Size'])
            neighborhood_quality = float(request.form['Neighborhood_Quality'])
            
            features = np.array([[
                square_footage, 
                num_bedrooms, 
                num_bathrooms, 
                year_built, 
                lot_size, 
                garage_size, 
                neighborhood_quality
            ]])
            
            prediction = model.predict(features)[0]
        except Exception as e:
            print(f"Error during prediction: {e}")
            
    return render_template_string(HTML_TEMPLATE, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
