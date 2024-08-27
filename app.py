import io
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, render_template, Response
import pandas as pd
import numpy as np

app = Flask(__name__)

# Define threshold levels for pollutants
thresholds = {
    'PM10': 50,
    'PM2.5': 25,
    'NO2': 40,
    'NH3': 35,
    'SO2': 20,
    'CO': 10,
    'O3': 100
}

# Load data from the local CSV file once at startup
data = pd.read_csv('cleaned_data.csv')

# Define AQI breakpoints and ranges
breakpoints = {
    'PM2.5': [(0, 12), (12.1, 35.4), (35.5, 55.4), (55.5, 150.4), (150.5, 250.4), (250.5, 350.4), (350.5, 500.4)],
    'PM10': [(0, 54), (55, 154), (155, 254), (255, 354), (355, 424), (425, 504), (505, 604)],
    'NO2': [(0, 53), (54, 100), (101, 360), (361, 649), (650, 1249), (1250, 1649), (1650, 2049)],
    'CO': [(0, 4.4), (4.5, 9.4), (9.5, 12.4), (12.5, 15.4), (15.5, 30.4), (30.5, 40.4), (40.5, 50.4)],
    'SO2': [(0, 35), (36, 75), (76, 185), (186, 304), (305, 604), (605, 804), (805, 1004)],
    'O3': [(0, 54), (55, 70), (71, 85), (86, 105), (106, 200), (201, 300), (301, 400)],
    'NH3': [(0, 200), (201, 400), (401, 800), (801, 1200), (1201, 1800), (1801, 2400), (2401, 3000)]
}

aqi_ranges = [(0, 50), (51, 100), (101, 150), (151, 200), (201, 300), (301, 400), (401, 500)]


# Function to calculate sub-index
def calculate_sub_index(concentration, breakpoints, aqi_ranges):
    for i, (bp_low, bp_high) in enumerate(breakpoints):
        if bp_low <= concentration <= bp_high:
            aqi_low, aqi_high = aqi_ranges[i]
            return ((concentration - bp_low) / (bp_high - bp_low)) * (aqi_high - aqi_low) + aqi_low
    return np.nan


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/states', methods=['GET'])
def get_states():
    states = data['state'].unique().tolist()
    return jsonify(states)


@app.route('/cities', methods=['GET'])
def get_cities():
    state = request.args.get('state')
    cities = data[data['state'] == state]['city'].unique().tolist()
    return jsonify(cities)


@app.route('/check_pollution', methods=['GET'])
def check_pollution():
    city = request.args.get('city')
    state = request.args.get('state')

    # Filter data for the given city and state
    place_data = data[(data['city'].str.lower() == city.lower()) & (data['state'] == state)]

    if place_data.empty:
        return jsonify({'error': 'City not found'}), 404

    # Check if any pollutant exceeds the threshold
    result = {}
    for pollutant, threshold in thresholds.items():
        max_value = place_data.loc[place_data['pollutant_id'] == pollutant, 'pollutant_max'].max()
        max_value = 0 if pd.isna(max_value) else max_value  # Handle NA values
        if max_value > threshold:
            result[pollutant] = 'Exceeded'
        else:
            result[pollutant] = 'Safe'

    return jsonify(result)


@app.route('/plot_pollution', methods=['GET'])
def plot_pollution():
    state = request.args.get('state')
    city = request.args.get('city')

    # Filter data for the selected state
    state_data = data[data['state'] == state]

    # Group and calculate AQI
    city_pollution = state_data.groupby(['city', 'pollutant_id'])['pollutant_avg'].mean().unstack()

    # Apply the sub-index calculation
    for pollutant in breakpoints.keys():
        if pollutant in city_pollution.columns:
            city_pollution[pollutant + '_AQI'] = city_pollution[pollutant].apply(calculate_sub_index, args=(
            breakpoints[pollutant], aqi_ranges))

    city_pollution['Composite_AQI'] = city_pollution[[col for col in city_pollution.columns if '_AQI' in col]].max(
        axis=1)

    # Create plot
    plt.figure(figsize=(15, 8))
    bars = plt.bar(city_pollution.index, city_pollution['Composite_AQI'], color='skyblue')

    # Highlight selected city and annotate it
    if city in city_pollution.index:
        selected_city_index = city_pollution.index.get_loc(city)
        bars[selected_city_index].set_color('red')
        plt.text(city, city_pollution.loc[city, 'Composite_AQI'], f'{city_pollution.loc[city, "Composite_AQI"]:.2f}',
                 ha='center', va='bottom', color='black', fontsize=12, fontweight='bold')

    plt.xlabel('City')
    plt.ylabel('Composite AQI')
    plt.title(f'Composite Pollution Index (AQI) by City in {state}')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # Save plot to a BytesIO object and return it
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()

    return Response(img, mimetype='image/png')


@app.route('/city_aqi', methods=['GET'])
def city_aqi():
    state = request.args.get('state')
    city = request.args.get('city')

    # Filter data for the selected state and city
    state_data = data[data['state'] == state]
    city_data = state_data[state_data['city'].str.lower() == city.lower()]

    if city_data.empty:
        return jsonify({'error': 'City not found'}), 404

    city_pollution = city_data.groupby('pollutant_id')['pollutant_avg'].mean()
    aqi_results = {}

    for pollutant in breakpoints.keys():
        if pollutant in city_pollution.index:
            aqi_results[pollutant] = calculate_sub_index(city_pollution[pollutant], breakpoints[pollutant], aqi_ranges)

    return jsonify(aqi_results)


@app.route('/state_aqi', methods=['GET'])
def state_aqi():
    state = request.args.get('state')

    # Filter data for the selected state
    state_data = data[data['state'] == state]

    if state_data.empty:
        return jsonify({'error': 'State not found'}), 404

    # Calculate AQI for each city in the state
    city_pollution = state_data.groupby(['city', 'pollutant_id'])['pollutant_avg'].mean().unstack()

    # Apply the sub-index calculation
    for pollutant in breakpoints.keys():
        if pollutant in city_pollution.columns:
            city_pollution[pollutant + '_AQI'] = city_pollution[pollutant].apply(calculate_sub_index, args=(
            breakpoints[pollutant], aqi_ranges))

    # Calculate the composite AQI for each city
    city_pollution['Composite_AQI'] = city_pollution[[col for col in city_pollution.columns if '_AQI' in col]].max(
        axis=1)

    # Calculate the overall AQI for the state
    state_aqi = city_pollution['Composite_AQI'].mean()

    return jsonify({'State_AQI': state_aqi})


@app.route('/city_and_state_aqi', methods=['GET'])
def city_and_state_aqi():
    state = request.args.get('state')
    city = request.args.get('city')

    # Filter data for the selected state
    state_data = data[data['state'] == state]
    city_data = state_data[state_data['city'].str.lower() == city.lower()]

    if city_data.empty:
        return jsonify({'error': 'City not found'}), 404

    # Calculate AQI for the selected city
    city_pollution = city_data.groupby('pollutant_id')['pollutant_avg'].mean()
    city_aqi_results = {}

    for pollutant in breakpoints.keys():
        if pollutant in city_pollution.index:
            city_aqi_results[pollutant] = calculate_sub_index(city_pollution[pollutant], breakpoints[pollutant], aqi_ranges)

    # Calculate AQI for the entire state
    city_pollution_all = state_data.groupby(['city', 'pollutant_id'])['pollutant_avg'].mean().unstack()

    for pollutant in breakpoints.keys():
        if pollutant in city_pollution_all.columns:
            city_pollution_all[pollutant + '_AQI'] = city_pollution_all[pollutant].apply(calculate_sub_index, args=(
            breakpoints[pollutant], aqi_ranges))

    city_pollution_all['Composite_AQI'] = city_pollution_all[[col for col in city_pollution_all.columns if '_AQI' in col]].max(
        axis=1)

    state_aqi = city_pollution_all['Composite_AQI'].mean()

    return jsonify({'City_AQI': city_aqi_results, 'State_AQI': state_aqi})


if __name__ == '__main__':
    app.run(debug=True)



