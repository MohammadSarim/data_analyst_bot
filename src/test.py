import pandas as pd
import numpy as np
import sqlite3
from faker import Faker
import random
from datetime import datetime, timedelta

# Initialize Faker
fake = Faker()
Faker.seed(42)
np.random.seed(42)

# Number of rows
n_rows = 10000

# Generate synthetic data
def generate_loyalty_data(n):
    data = []
    campaigns = ['Spring Saver', 'Double Miles', 'Festive Bonanza', 'Referral Boost', 'Premium Perks']
    travel_classes = ['Eco', 'Business']
    travel_types = ['Personal Travel', 'Business Travel']
    satisfaction_levels = ['satisfied', 'neutral', 'dissatisfied']
    
    for _ in range(n):
        join_date = fake.date_between(start_date='-2y', end_date='-1m')
        flight_date = join_date + timedelta(days=np.random.randint(1, 365))
        campaign = random.choice(campaigns)
        gender = random.choice(['Male', 'Female'])
        customer_type = random.choice(['Loyal Customer', 'New Customer'])
        age = np.random.randint(18, 80)
        travel_type = random.choice(travel_types)
        travel_class = random.choice(travel_classes)
        distance = np.random.randint(200, 12000)
        satisfaction = random.choice(satisfaction_levels)
        metrics = {col: np.random.randint(0, 6) for col in [
            'Seat_comfort', 'Departure_convenience', 'Food_and_drink', 'Gate_location',
            'Inflight_wifi', 'Inflight_entertainment', 'Online_support', 'Ease_of_booking',
            'Onboard_service', 'Leg_room', 'Baggage_handling', 'Checkin_service',
            'Cleanliness', 'Online_boarding'
        ]}
        dep_delay = np.random.choice([0, np.random.randint(10, 300)], p=[0.7, 0.3])
        arr_delay = dep_delay + np.random.randint(-10, 30)

        data.append({
            'customer_id': fake.uuid4(),
            'gender': gender,
            'customer_type': customer_type,
            'age': age,
            'travel_type': travel_type,
            'travel_class': travel_class,
            'flight_distance': distance,
            'campaign_name': campaign,
            'flight_date': flight_date,
            'join_date': join_date,
            'satisfaction': satisfaction,
            'departure_delay': dep_delay,
            'arrival_delay': arr_delay,
            **metrics
        })
    return pd.DataFrame(data)

# Create DataFrame
df = generate_loyalty_data(n_rows)

# Save to SQLite
db_path = "airline.csv"
conn = sqlite3.connect(db_path)
df.to_sql("airline", conn, if_exists="replace", index=False)
conn.close()

