import paho.mqtt.client as mqtt
import msgpack
import time
import os
import random
from datetime import datetime

BROKER = "localhost"
TOPIC = "masa/camera/data"

def send_vehicle(client, v_id):
    classes = ["car", "bus", "motorbike", "truck"]

    lat_base, lon_base = 44.648, 10.920
    lat = round(lat_base + random.uniform(-0.0001, 0.0001), 6)
    lon = round(lon_base + random.uniform(-0.0001, 0.0001), 6)

    data = {
        "track_id": v_id,
        "class": random.choice(classes),
        "lat": lat,
        "long": lon,
        "timestamp": time.time()
    }

    client.publish(TOPIC, msgpack.packb(data))
    ora_invio = datetime.fromtimestamp(data['timestamp']).strftime('%H:%M:%S')
    print(f"[{ora_invio}] INVIATO: ID={v_id} | GPS={lat},{lon} | Tipo={data['class']}")

def run_simulation():
    client = mqtt.Client(client_id="Camera_Sensor_Stocastico")
    client.connect(BROKER, 1883)

    id_counter = 1
    is_rush = False
    next_change = time.time()

    print(f"--- SIMULATORE MASA: GENERAZIONE DATI ATTIVA ---")

    try:
        while True:
            now = time.time()

            # Logica di cambio scenario "casuale"
            if now >= next_change:
                is_rush = not is_rush

                if is_rush:
                    durata_fase = random.uniform(10, 15)
                    status_msg = "Congestione (Probabilità 80%)"
                else:
                    durata_fase = random.uniform(20, 30)
                    status_msg = "Regolare (Probabilità 15%)"

                next_change = now + durata_fase
                print(f"\n>>> CAMBIO SCENARIO: {status_msg} per {int(durata_fase)}s")

            # GENERAZIONE TRAFFICO
            if is_rush:
                # Controllo ogni 0.7 secondi durante la congestione
                if random.random() < 0.8:
                    id_counter += 1
                    send_vehicle(client, id_counter)
                time.sleep(0.7)
            else:
                # Controllo ogni 1.0 secondo durante il traffico regolare
                if random.random() < 0.15:
                    id_counter += 1
                    send_vehicle(client, id_counter)
                time.sleep(1.0)

    except KeyboardInterrupt:
        print("\nSpegnimento simulatore.")
        os._exit(0)

if __name__ == "__main__":
    run_simulation()