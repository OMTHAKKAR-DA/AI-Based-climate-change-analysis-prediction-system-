import csv
import random
import math
from datetime import datetime, timedelta

def generate_climate_data(num_days=365*5, output_file='data/climate_data.csv'):
    """
    Generates synthetic climate data for 5 years using only standard libraries.
    """
    start_date = datetime(2018, 1, 1)
    
    random.seed(42)
    
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Date', 'Temperature_C', 'Humidity_pct', 'Rainfall_mm', 'AQI'])
        
        for i in range(num_days):
            current_date = start_date + timedelta(days=i)
            time_index = i
            days_in_year = 365.25
            
            # Temperature
            base_temp = 25
            yearly_seasonality = 10 * math.sin(2 * math.pi * time_index / days_in_year - math.pi/2)
            trend = 1.0 * (time_index / num_days)
            noise_temp = random.gauss(0, 2)
            temperature = round(base_temp + yearly_seasonality + trend + noise_temp, 2)
            
            # Humidity
            base_humidity = 60
            yearly_humidity = 15 * math.cos(2 * math.pi * time_index / days_in_year - math.pi/2)
            noise_humidity = random.gauss(0, 5)
            humidity = round(max(10, min(100, base_humidity + yearly_humidity + noise_humidity)), 2)
            
            # Rainfall
            rain_probability = 0.2 + 0.3 * max(0, min(1, math.sin(2 * math.pi * time_index / days_in_year - math.pi)))
            rain_event = 1 if random.random() < rain_probability else 0
            rain_amount = round(random.expovariate(1/15) * rain_event, 2) if rain_event else 0.0
            
            # AQI
            base_aqi = 50
            aqi_seasonality = 40 * math.cos(2 * math.pi * time_index / days_in_year)
            aqi_trend = 20 * (time_index / num_days)
            noise_aqi = random.gauss(0, 10)
            aqi = int(max(0, min(500, base_aqi + aqi_seasonality + aqi_trend + noise_aqi)))
            
            writer.writerow([current_date.strftime('%Y-%m-%d'), temperature, humidity, rain_amount, aqi])
            
    print(f"Successfully generated {num_days} records and saved to {output_file}")

if __name__ == "__main__":
    generate_climate_data()
