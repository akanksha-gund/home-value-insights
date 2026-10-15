from flask import Flask, render_template_string, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained linear regression model
with open('linear.pkl', 'rb') as f:
    model = pickle.load(f)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Advanced House Price Predictor</title>
    <!-- Google Fonts & Font Awesome Icons -->
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    
    <style>
        :root {
            --primary: #4f46e5;
            --primary-hover: #4338ca;
            --bg-color: #f8fafc;
            --card-bg: #ffffff;
            --text-main: #0f172a;
            --text-muted: #64748b;
            --border: #e2e8f0;
            --input-bg: #f8fafc;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: 'Plus Jakarta Sans', sans-serif;
            background-color: var(--bg-color);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 30px 15px;
        }

        .wrapper {
            background: var(--card-bg);
            border-radius: 20px;
            box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.07);
            max-width: 720px;
            width: 100%;
            overflow: hidden;
            border: 1px solid var(--border);
        }

        /* Top Hero Header with Background Image */
        .hero-banner {
            position: relative;
            height: 180px;
            background: linear-gradient(180deg, rgba(15, 23, 42, 0.4) 0%, rgba(15, 23, 42, 0.8) 100%), 
                        url('https://images.unsplash.com/photo-1600596542815-ffad4c1539a9?auto=format&fit=crop&w=1200&q=80') center/cover no-repeat;
            display: flex;
            flex-direction: column;
            justify-content: flex-end;
            padding: 25px 35px;
            color: #ffffff;
        }

        .hero-banner h1 {
            font-size: 26px;
            font-weight: 700;
            letter-spacing: -0.5px;
        }

        .hero-banner p {
            font-size: 14px;
            color: #cbd5e1;
            margin-top: 4px;
        }

        .form-container {
            padding: 35px;
        }

        .form-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 20px;
        }

        .form-group {
            display: flex;
            flex-direction: column;
        }

        .form-group.full-width {
            grid-column: span 2;
        }

        label {
            font-size: 13px;
            font-weight: 600;
            color: var(--text-main);
            margin-bottom: 8px;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        label i {
            color: var(--primary);
            font-size: 14px;
            width: 16px;
        }

        .input-wrapper {
            position: relative;
        }

        input {
            width: 100%;
            padding: 12px 14px;
            border: 1px solid var(--border);
            border-radius: 10px;
            background-color: var(--input-bg);
            font-size: 14px;
            font-family: inherit;
            color: var(--text-main);
            transition: all 0.2s ease;
        }

        input:focus {
            outline: none;
            background-color: #ffffff;
            border-color: var(--primary);
            box-shadow: 0 0 0 4px rgba(79, 70, 229, 0.1);
        }

        button.btn-submit {
            grid-column: span 2;
            margin-top: 10px;
            width: 100%;
            padding: 14px;
            background-color: var(--primary);
            color: white;
            border: none;
            border-radius: 10px;
            font-size: 15px;
            font-weight: 600;
            cursor: pointer;
            transition: background-color 0.2s ease, transform 0.1s ease;
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 8px;
        }

        button.btn-submit:hover {
            background-color: var(--primary-hover);
        }

        button.btn-submit:active {
            transform: scale(0.99);
        }

        /* Prediction Result Box */
        .result-card {
            margin-top: 25px;
            padding: 20px 25px;
            background: linear-gradient(135deg, #eef2ff 0%, #e0e7ff 100%);
            border: 1px solid #c7d2fe;
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .result-info h3 {
            font-size: 13px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: #4338ca;
            font-weight: 700;
        }

        .result-info p {
            font-size: 12px;
            color: #6366f1;
            margin-top: 2px;
        }

        .result-price {
            font-size: 28px;
            font-weight: 800;
            color: #3730a3;
        }

        @media (max-width: 640px) {
            .form-grid {
                grid-template-columns: 1fr;
            }
            .form-group.full-width, button.btn-submit {
                grid-column: span 1;
            }
            .result-card {
                flex-direction: column;
                align-items: flex-start;
                gap: 10px;
            }
        }
    </style>
</head>
<body>

    <div class="wrapper">
        <!-- Modern Home Banner Image -->
        <div class="hero-banner">
            <h1>House Price Predictor</h1>
            <p>Smart real estate valuation powered by Machine Learning</p>
        </div>

        <div class="form-container">
            <form method="POST">
                <div class="form-grid">
                    
                    <div class="form-group">
                        <label for="Square_Footage"><i class="fa-solid fa-ruler-combined"></i> Square Footage</label>
                        <input type="number" step="any" id="Square_Footage" name="Square_Footage" placeholder="e.g., 2000" required>
                    </div>

                    <div class="form-group">
                        <label for="Num_Bedrooms"><i class="fa-solid fa-bed"></i> Bedrooms</label>
                        <input type="number" id="Num_Bedrooms" name="Num_Bedrooms" placeholder="e.g., 3" required>
                    </div>

                    <div class="form-group">
                        <label for="Num_Bathrooms"><i class="fa-solid fa-bath"></i> Bathrooms</label>
                        <input type="number" step="any" id="Num_Bathrooms" name="Num_Bathrooms" placeholder="e.g., 2" required>
                    </div>

                    <div class="form-group">
                        <label for="Year_Built"><i class="fa-solid fa-calendar-days"></i> Year Built</label>
                        <input type="number" id="Year_Built" name="Year_Built" placeholder="e.g., 2010" required>
                    </div>

                    <div class="form-group">
                        <label for="Lot_Size"><i class="fa-solid fa-vector-square"></i> Lot Size (sq ft)</label>
                        <input type="number" step="any" id="Lot_Size" name="Lot_Size" placeholder="e.g., 5000" required>
                    </div>

                    <div class="form-group">
                        <label for="Garage_Size"><i class="fa-solid fa-car"></i> Garage Capacity</label>
                        <input type="number" step="any" id="Garage_Size" name="Garage_Size" placeholder="e.g., 2 cars" required>
                    </div>

                    <div class="form-group full-width">
                        <label for="Neighborhood_Quality"><i class="fa-solid fa-star"></i> Neighborhood Quality (1-10)</label>
                        <input type="number" step="any" id="Neighborhood_Quality" name="Neighborhood_Quality" placeholder="e.g., 8" required>
                    </div>

                    <button type="submit" class="btn-submit">
                        <i class="fa-solid fa-calculator"></i> Calculate Estimate
                    </button>
                </div>
            </form>

            {% if prediction is not none %}
            <div class="result-card">
                <div class="result-info">
                    <h3>Estimated Market Value</h3>
                    <p>Based on inputted property parameters</p>
                </div>
                <div class="result-price">
                    ${{ "{:,.2f}".format(prediction) }}
                </div>
            </div>
            {% endif %}
        </div>
    </div>

</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    if request.method == 'POST':
        try:
            # Extract inputs in precise model order
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
            print(f"Prediction Error: {e}")
            
    return render_template_string(HTML_TEMPLATE, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
