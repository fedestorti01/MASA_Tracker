import paho.mqtt.client as mqtt
import msgpack
import time
import random

client = mqtt.Client()
client.connect("localhost", 1883)


def send_mock_data(v_id):
    t_now = time.time()
    payload = {
        "id": v_id,
        "cls": random.choice(["car", "motorcycle"]),
        "lat": 44.654818,
        "lon": 10.934699,
        "t_detection": time.strftime("%H:%M:%S", time.localtime(t_now)),
        "t_send": t_now
    }
    client.publish("masa/sim/data", msgpack.packb(payload))


print("--- START SIMULAZIONE (90s) ---")
start = time.time()
v_id_counter = 500

try:
    while time.time() - start < 90:
        current_elapsed = time.time() - start

        # FASE 1 & 3: Traffico scarso (Ogni 7s)
        if current_elapsed < 30 or current_elapsed > 65:
            send_mock_data(v_id_counter)
            v_id_counter += 1
            time.sleep(7)

        # FASE 2: Traffico intenso (Ogni 1.5s) -> Attiva PRIORITY
        else:
            send_mock_data(v_id_counter)
            v_id_counter += 1
            time.sleep(1.5)
except KeyboardInterrupt:
    pass

client.disconnect()
print("Simulazione terminata.")