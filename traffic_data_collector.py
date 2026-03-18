import paho.mqtt.client as mqtt
import msgpack
import time
from datetime import datetime

# Configurazione
BROKER = "localhost"
SENSOR_TOPIC = "masa/camera/data"
TL_ID = "Traffic Light 01"
COMMAND_TOPIC = f"masa/infrastructure/trafficlight/{TL_ID}/command"

# Memoria per il calcolo della densità
vehicle_history = []

def on_message(client, userdata, msg):
    global vehicle_history
    try:
        # Decodifica il pacchetto Msgpack
        data = msgpack.unpackb(msg.payload)

        ts_originale = data.get('timestamp')
        v_id = data.get('track_id')
        v_class = data.get('class', 'N/D')
        v_lat = data.get('lat')
        v_long = data.get('long')

        ora_rilevamento = datetime.fromtimestamp(ts_originale).strftime('%H:%M:%S') if ts_originale else "N/D"

        # Stampa delle informazioni ricevute dalla camera sintetica
        print(f"[{ora_rilevamento}] RICEVUTO: ID={v_id} | Tipo={v_class} | GPS={v_lat},{v_long}")

        # Registriamo l'evento per la logica di priorità
        vehicle_history.append((time.time(), v_id))

    except Exception as e:
        print(f"Errore durante la ricezione: {e}")

def monitor_traffic():
    client = mqtt.Client(client_id="Traffic_Data_Collector")
    client.on_message = on_message
    client.connect(BROKER, 1883)
    client.subscribe(SENSOR_TOPIC)
    client.loop_start()

    print(f"COLLECTOR ATTIVO - Monitoraggio {TL_ID}...")
    print(f"Soglia Priority: >= 6 veicoli univoci in 10 secondi\n")

    last_status = "NORMAL"

    try:
        while True:
            current_time = time.time()

            # Sliding window 10 secondi
            vehicle_history[:] = [v for v in vehicle_history if current_time - v[0] <= 10]

            unique_vehicles = len(set(v[1] for v in vehicle_history))

            # Logica decisionale
            new_status = "PRIORITY" if unique_vehicles >= 6 else "NORMAL"

            # Comando switch stato
            if new_status != last_status:
                client.publish(COMMAND_TOPIC, new_status)
                ora_attuale = datetime.now().strftime('%H:%M:%S')
                print(f"\n--- [{ora_attuale}] CAMBIO STATO: {new_status} ---")
                last_status = new_status
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\nSpegnimento Collector.")
        client.loop_stop()

if __name__ == "__main__":
    monitor_traffic()