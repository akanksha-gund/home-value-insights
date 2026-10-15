from flask import Flask, render_template_string, request
import numpy as np
import pickle

app = Flask(__name__)

# Model load karat ahot
with open('linear.pkl', 'rb') as f:
  model = pickle.load(f)

# HTML template simple ani clean thevla ahe
html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>House Price Prediction</title>
    <style>
        body { font-family: Arial, sans-serif; background-color: #f4f4f9; margin: 0; padding: 50px; }
        .container { max-width: 500px; background: white; padding: 30px; border-radius: 8px; box-shadow: 0px 0px 10px rgba(0,0,0,0.1); margin: auto; }
        h2 { text-align: center; color: #333; }
        .form-group { margin-bottom: 15px; }
        label { display: block; margin-bottom: 5px; color: #666; font-weight: bold; }
        input { width: 100%; padding: 8px; box-sizing: border-box; border: 1px solid #ccc; border-radius: 4px; }
        button { width: 100%; background-color: #28a745; color: white; padding: 10px; border: none; border-radius: 4px; font-size: 16px; cursor: pointer; }
        button:hover { background-color: #218838; }
        .result { margin-top: 20px; text-align: center; font-size: 20px; color: #333; }
    </style>
</head>
<body>
    <div class="container">
        <h2>House Price Predictor</h2>
        <form method="POST">
            <div class="form-group">
                <label>Square Footage:</label>
                <input type="number" step="any" name="Square_Footage" required>
            </div>
            <div class="form-group">
                <label>Num Bedrooms:</label>
                <input type="number" name="Num_Bedrooms" required>
            </div>
            <div class="form-group">
                <label>Num Bathrooms:</label>
                <input type="number" step="any" name="Num_Bathrooms" required>
            </div>
            <div class="form-group">
                <label>Year Built:</label>
                <input type="number" name="Year_Built" required>
            </div>
            <div class="form-group">
                <label>Lot Size:</label>
                <input type="number" step="any" name="Lot_Size" required>
            </div>
            <div class="form-group">
                <label>Garage Size:</label>
                <input type="number" name="Garage_Size" required>
            </div>
            <div class="form-group">
                <label>Neighborhood Quality:</label>
                <input type="number" step="any" name="Neighborhood_Quality" required>
            </div>
            <button type="submit">Predict Price</button>
        </form>
        {% if prediction is not none %}
        <div class="result">
            <h3>Predicted Price: ${{ prediction }}</h3>
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
      # Form madhun values ghet ahot
      sq_ft = float(request.form['Square_Footage'])
      bedrooms = float(request.form['Num_Bedrooms'])
      bathrooms = float(request.form['Num_Bathrooms'])
      year_built = float(request.form['Year_Built'])
      lot_size = float(request.form['Lot_Size'])
      garage_size = float(request.form['Garage_Size'])
      neighborhood_quality = float(request.form['Neighborhood_Quality'])

      # Model sathi array tayar karat ahot
      features = np.array([[
          sq_ft,
          bedrooms,
          bathrooms,
          year_built,
          lot_size,
          garage_size,
          neighborhood_quality,
      ]])

      # Prediction
      pred = model.predict(features)[0]
      prediction = round(pred, 2)
    except Exception as e:
      prediction = f'Error: {str(e)}'

  return render_template_string(html_template, prediction=prediction)


if __name__ == '__main__':
  app.run(debug=True)
