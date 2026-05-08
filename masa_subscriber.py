import paho.mqtt.client as mqtt
import msgpack
import time

def on_message(client, userdata, msg):
    t_arrival = time.time()
    try:
        data = msgpack.unpackb(msg.payload)

        # Messaggi di Health Check
        if "health" in msg.topic:
            print(
                f"[HEALTH] CPU: {data['cpu_percent']}% | RAM: {data['memory_percent']}% | Buffer: {data['buffer_usage']}")

        # Messaggi di Status (Online/Offline)
        elif "status" in msg.topic:
            print(f"[STATUS] Il sistema è: {data['status']}")

        # Messaggi di Tracking
        else:
            if "t_send" in data:
                latency = (t_arrival - data['t_send']) * 1000
                print(f"DATO RICEVUTO | ID:{data['id']} | Classe:{data['cls']} | Coord: {data['lat']} | {data['lon']} | ROI: {data['ROI']} | Latenza: {latency:.2f} ms | Payload: {len(msg.payload)} bytes")

    except Exception as e:
        print(f"Errore nella decodifica sul topic {msg.topic}: {e}")

    def on_message(client, userdata, msg):
        try:
            data = msgpack.unpackb(msg.payload)

            if "health" in msg.topic:
                print(
                    f"[HEALTH] CPU: {data['cpu_percent']}% | RAM: {data['memory_percent']}% | Buffer: {data['buffer_usage']}")
            elif "system/status" in msg.topic:
                print(f"[STATUS] Il sistema è: {data['status']}")
            else:
                print(f"[DATA] Ricevuto tracking per topic: {msg.topic}")

        except Exception as e:
            print(f"Errore decodifica: {e}")

client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
client.on_connect = lambda c,u,f,rc,p: client.subscribe("masa/#")
client.on_message = on_message

print(f"SUBSCRIBER ATTIVO - Attesi dati da MASATracker...")
client.connect("localhost", 1883)
client.loop_forever()

