import pandas as pd
import folium
from geopy.distance import geodesic
from flask import Flask, render_template, request, send_file
import os
import uuid

# Function to read store coordinates
def read_store_coordinates(file_path):
    df = pd.read_csv(file_path)
    return df.iloc[0]['Latitude'], df.iloc[0]['Longitude']

# Function to read shipment data
def read_shipments(file_path):
    return pd.read_csv(file_path)

# Function to read vehicle data
def read_vehicles(file_path):
    return pd.read_csv(file_path)

# Function to calculate distance between two coordinates
def calculate_distance(lat1, lon1, lat2, lon2):
    return geodesic((lat1, lon1), (lat2, lon2)).km

# Function to find the farthest shipment from the origin
def find_farthest_shipment(df, origin):
    max_distance = -1
    farthest_shipment = None
    for _, row in df.iterrows():
        dist = calculate_distance(origin[0], origin[1], row["Latitude"], row["Longitude"])
        if dist > max_distance:
            max_distance = dist
            farthest_shipment = row
    return farthest_shipment

# Function to find the nearest shipment to a given shipment
def find_nearest_shipment(df, current_shipment):
    min_distance = float('inf')
    nearest_shipment = None
    for _, row in df.iterrows():
        dist = calculate_distance(current_shipment['Latitude'], current_shipment['Longitude'],
                                  row["Latitude"], row["Longitude"])
        if dist < min_distance and row['Shipment ID'] != current_shipment['Shipment ID']:
            min_distance = dist
            nearest_shipment = row
    return nearest_shipment

# Function to get vehicle type based on number of shipments
def get_vehicle_type(num_shipments, vehicles_df):
    for _, row in vehicles_df.iterrows():
        if row['Min_Shipments'] <= num_shipments <= row['Max_Shipments']:
            return row['Vehicle_Type']
    return "Unknown"

# Function to get capacity utilization
def get_capacity_uti(num_shipments, vehicle_type, vehicles_df):
    vehicle = vehicles_df[vehicles_df['Vehicle_Type'] == vehicle_type].iloc[0]
    return num_shipments / vehicle['Max_Shipments']

# Function to get time utilization
def get_time_uti(trip_time, time_slot):
    start_time, end_time = time_slot.split('-')
    start_time = pd.to_datetime(start_time, format='%H:%M:%S')
    end_time = pd.to_datetime(end_time, format='%H:%M:%S')
    time_slot_duration = (end_time - start_time).total_seconds() / 60  # in minutes
    return trip_time / time_slot_duration

# Function to process shipments using a specific strategy
def process_shipments(df_group, vehicles_df, max_shipments, strategy='farthest_first'):
    group_trips = {}
    for time_slot in df_group["Delivery Timeslot"].unique():
        df_time_slot = df_group[df_group["Delivery Timeslot"] == time_slot]
        visited_shipment_ids = set()
        trips = []

        if strategy == 'farthest_first':
            # Logic for 0-10 km (farthest first)
            while len(df_time_slot) > 0 and len(trips) < 50:  # Limit to 50 trips
                current_trip = []
                farthest_shipment = find_farthest_shipment(df_time_slot, origin)
                current_trip.append(farthest_shipment)
                visited_shipment_ids.add(farthest_shipment['Shipment ID'])
                df_time_slot = df_time_slot[df_time_slot['Shipment ID'] != farthest_shipment['Shipment ID']]
                current_shipment = farthest_shipment

                for _ in range(max_shipments - 1):
                    if len(df_time_slot) == 0:
                        break
                    nearest_shipment = find_nearest_shipment(df_time_slot, current_shipment)
                    if nearest_shipment is None:
                        break
                    current_trip.append(nearest_shipment)
                    visited_shipment_ids.add(nearest_shipment['Shipment ID'])
                    df_time_slot = df_time_slot[df_time_slot['Shipment ID'] != nearest_shipment['Shipment ID']]
                    current_shipment = nearest_shipment

                trips.append(current_trip)

        elif strategy == 'nearest_neighbor':
            # Logic for 10-20 km and remaining shipments (nearest neighbor)
            while len(df_time_slot) > 0:
                current_trip = []
                current_location = origin
                remaining_capacity = max_shipments

                while remaining_capacity > 0 and len(df_time_slot) > 0:
                    min_dist = float('inf')
                    nearest = None
                    for _, row in df_time_slot.iterrows():
                        if row['Shipment ID'] in visited_shipment_ids:
                            continue
                        dist = calculate_distance(current_location[0], current_location[1], row['Latitude'], row['Longitude'])
                        if dist < min_dist:
                            min_dist = dist
                            nearest = row

                    if nearest is None:
                        break  # No more shipments to add
                    current_trip.append(nearest)
                    visited_shipment_ids.add(nearest['Shipment ID'])
                    df_time_slot = df_time_slot[df_time_slot['Shipment ID'] != nearest['Shipment ID']]
                    current_location = (nearest['Latitude'], nearest['Longitude'])
                    remaining_capacity -= 1

                if current_trip:
                    trips.append(current_trip)  # Add the trip even if it has fewer than max_shipments

        group_trips[time_slot] = trips
    return group_trips

# Function to add trips to the map with specific colors and popups
def add_trips_to_map(trips, layer_name, color_scheme, time_slot, vehicles_df):
    layer = folium.FeatureGroup(name=layer_name)
    for idx, trip in enumerate(trips):
        color = color_scheme[idx % len(color_scheme)]
        path = [origin]
        shipment_ids = []
        shipment_coords = []
        
        for shipment in trip:
            lat = shipment['Latitude']
            lon = shipment['Longitude']
            shipment_coords.append((lat, lon))
            shipment_ids.append(str(shipment['Shipment ID']))
            path.append((lat, lon))
        
        # Calculate total distance including return to origin
        total_distance = 0.0
        for i in range(len(path) - 1):
            total_distance += geodesic(path[i], path[i+1]).km
        return_distance = geodesic(path[-1], origin).km
        total_distance += return_distance
        
        # Calculate trip time
        num_deliveries = len(trip)
        trip_time = 5 * total_distance + 10 * num_deliveries
        
        # Determine vehicle type
        vehicle_type = get_vehicle_type(num_deliveries, vehicles_df)
        
        # Calculate capacity utilization
        capacity_uti = get_capacity_uti(num_deliveries, vehicle_type, vehicles_df)
        
        # Calculate time utilization
        time_uti = get_time_uti(trip_time, time_slot)
        
        # Set coverage utilization
        cov_uti = 1
        
        # Create popup content
        popup_content = f"""
        <b>Total Distance:</b> {total_distance:.2f} km<br>
        <b>Shipment IDs:</b> {', '.join(shipment_ids)}<br>
        <b>Trip Time:</b> {trip_time:.2f} minutes<br>
        <b>Vehicle Type:</b> {vehicle_type}<br>
        <b>Capacity Utilization:</b> {capacity_uti:.2f}<br>
        <b>Time Utilization:</b> {time_uti:.2f}<br>
        <b>Coverage Utilization:</b> {cov_uti:.2f}
        """
        popup = folium.Popup(popup_content, max_width=300)
        
        # Add the PolyLine with popup
        folium.PolyLine(
            locations=path,
            color=color,
            weight=2.5,
            opacity=0.7,
            popup=popup
        ).add_to(layer)
        
        # Add dashed line back to origin
        folium.PolyLine(
            locations=[path[-1], origin],
            color='black',
            weight=2,
            opacity=0.7,
            dash_array='5,5'
        ).add_to(layer)
        
        # Add CircleMarkers for each shipment
        for coord, ship_id in zip(shipment_coords, shipment_ids):
            folium.CircleMarker(
                location=coord,
                radius=8,
                color=color,
                fill=True,
                popup=f"Shipment {ship_id}"
            ).add_to(layer)
    
    layer.add_to(m)

# Function to generate the output CSV
def generate_output_csv(time_slot_trips, output_file, vehicles_df):
    trip_data = []
    for time_slot, trips in time_slot_trips.items():
        for trip_idx, trip in enumerate(trips):
            shipment_ids = [shipment['Shipment ID'] for shipment in trip]
            latitudes = [shipment['Latitude'] for shipment in trip]
            longitudes = [shipment['Longitude'] for shipment in trip]

            total_distance = 0.0
            path = [origin] + [(lat, lon) for lat, lon in zip(latitudes, longitudes)]
            for i in range(len(path) - 1):
                total_distance += geodesic(path[i], path[i + 1]).km
            return_distance = geodesic(path[-1], origin).km
            total_distance += return_distance

            num_deliveries = len(trip)
            trip_time = 5 * total_distance + 10 * num_deliveries

            vehicle_type = get_vehicle_type(num_deliveries, vehicles_df)
            capacity_uti = get_capacity_uti(num_deliveries, vehicle_type, vehicles_df)
            time_uti = get_time_uti(trip_time, time_slot)
            cov_uti = 1

            for i in range(len(shipment_ids)):
                trip_data.append({
                    "TRIP ID": f"Trip_{trip_idx + 1}_{time_slot}",
                    "Shipment ID": shipment_ids[i],
                    "Latitude": latitudes[i],
                    "Longitude": longitudes[i],
                    "TIME SLOT": time_slot,
                    "Shipments": num_deliveries,
                    "MST_DIST": total_distance,
                    "TRIP_TIME": trip_time,
                    "Vehical_Type": vehicle_type,
                    "CAPACITY_UTI": capacity_uti,
                    "TIME_UTI": time_uti,
                    "COV_UTI": cov_uti
                })

    df_output = pd.DataFrame(trip_data)
    df_output.to_csv(output_file, index=False)
    print(f"Output saved to {output_file}")

# Flask App
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route('/', methods=['GET'])
def index():
    return render_template('main.html')

@app.route('/process', methods=['POST'])
def process_files():
    try:
        session_id = str(uuid.uuid4())
        os.makedirs(f'{UPLOAD_FOLDER}/{session_id}', exist_ok=True)
        os.makedirs(f'{OUTPUT_FOLDER}/{session_id}', exist_ok=True)

        store_file = request.files['store_file']
        shipment_file = request.files['shipment_file']
        vehicle_file = request.files['vehicle_file']
        
        store_path = f'{UPLOAD_FOLDER}/{session_id}/store.csv'
        shipment_path = f'{UPLOAD_FOLDER}/{session_id}/shipments.csv'
        vehicle_path = f'{UPLOAD_FOLDER}/{session_id}/vehicles.csv'
        
        store_file.save(store_path)
        shipment_file.save(shipment_path)
        vehicle_file.save(vehicle_path)

        origin = read_store_coordinates(store_path)
        df = read_shipments(shipment_path)
        vehicles_df = read_vehicles(vehicle_path)

        df['Distance (km)'] = df.apply(lambda row: calculate_distance(origin[0], origin[1], row["Latitude"], row["Longitude"]), axis=1)
        df_within_10km = df[df['Distance (km)'] <= 10]
        df_10_20km = df[(df['Distance (km)'] > 10) & (df['Distance (km)'] <= 20)]

        time_slot_trips_10km = process_shipments(df_within_10km, vehicles_df, max_shipments=5, strategy='farthest_first')
        remaining_shipments_10km = df_within_10km[~df_within_10km['Shipment ID'].isin(
            [shipment['Shipment ID'] for trips in time_slot_trips_10km.values() for trip in trips for shipment in trip]
        )]

        remaining_shipments = pd.concat([remaining_shipments_10km, df_10_20km])
        time_slot_trips_remaining = {}
        for time_slot in remaining_shipments["Delivery Timeslot"].unique():
            max_shipments = 13 if time_slot == "09:30:00-12:00:00" else 8
            df_time_slot = remaining_shipments[remaining_shipments["Delivery Timeslot"] == time_slot]
            time_slot_trips_remaining[time_slot] = process_shipments(df_time_slot, vehicles_df, max_shipments=max_shipments, strategy='nearest_neighbor')[time_slot]

        time_slot_trips = {}
        for time_slot in time_slot_trips_10km:
            time_slot_trips[time_slot] = time_slot_trips_10km[time_slot] + time_slot_trips_remaining.get(time_slot, [])

        # Create map
        global m
        m = folium.Map(location=origin, zoom_start=12)
        folium.Marker(origin, popup="Warehouse", icon=folium.Icon(color="black", icon="home")).add_to(m)
        folium.Circle(origin, radius=10000, color="blue", fill=True, fill_opacity=0.2, popup="10 km").add_to(m)
        folium.Circle(origin, radius=20000, color="red", fill=True, fill_opacity=0.2, popup="20 km").add_to(m)

        colors_10km = ['blue', 'green', 'red', 'orange', 'purple', 'cyan']
        colors_remaining = ['yellow', 'lime', 'magenta', 'teal', 'olive', 'navy']

        for time_slot, trips in time_slot_trips_10km.items():
            add_trips_to_map(trips, f"0-10 km: {time_slot}", colors_10km, time_slot, vehicles_df)

        for time_slot, trips in time_slot_trips_remaining.items():
            add_trips_to_map(trips, f"Remaining 0-20 km: {time_slot}", colors_remaining, time_slot, vehicles_df)

        folium.LayerControl().add_to(m)

        output_csv_path = f'{OUTPUT_FOLDER}/{session_id}/output_trips.csv'
        map_html_path = f'{OUTPUT_FOLDER}/{session_id}/map.html'
        
        generate_output_csv(time_slot_trips, output_csv_path, vehicles_df)
        m.save(map_html_path)

        return {
            'status': 'success',
            'session_id': session_id,
            'downloads': {
                'csv': f'/download/{session_id}/output_trips.csv',
                'map': f'/download/{session_id}/map.html'
            }
        }
    except Exception as e:
        return {'status': 'error', 'message': str(e)}

@app.route('/download/<path:filename>')
def download_file(filename):
    return send_file(f'{OUTPUT_FOLDER}/{filename}', as_attachment=True)

if __name__ == '_main_':
    app.run(host='0.0.0.0', port=5000, debug=True)