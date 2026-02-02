import paho.mqtt.client as mqtt
import msgpack
import time
import psutil
import threading
from collections import deque


class MASACommunication:
    def __init__(self, broker="localhost", port=1883, max_buffer_size=10000):
        self.client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
        self.broker = broker
        self.port = port
        self.connected = False

        # Buffer locale per la resilienza offline (deque con limite RAM)
        self.buffer = deque(maxlen=max_buffer_size)

        # Evento e Thread per l'Health Check
        self.stop_health_check = threading.Event()
        self.health_thread = None

        # Configurazione Last Will and Testament (LWT), se lo script crasha, il broker invia questo messaggio automatico
        lwt_payload = msgpack.packb({"status": "OFFLINE", "reason": "unexpected_crash"}, use_bin_type=True)
        self.client.will_set("masa/system/status", payload=lwt_payload, qos=2, retain=True)

        # Callbacks
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect

    def _on_connect(self, client, userdata, flags, rc, properties=None):
        if rc == 0:
            print("MQTT: Connessione stabilita.")
            self.connected = True

            # Notifica lo stato ONLINE
            self.send_system_event("system", "status", {"status": "ONLINE"}, qos=2)

            # Svuota il buffer dei dati accumulati durante l'offline
            self._flush_buffer()

            # Avvia il monitoraggio delle risorse in un thread separato
            if self.health_thread is None or not self.health_thread.is_alive():
                self.stop_health_check.clear()
                self.health_thread = threading.Thread(target=self._health_check_loop, daemon=True)
                self.health_thread.start()
        else:
            print(f"MQTT: Errore connessione (Codice: {rc})")

    def _on_disconnect(self, client, userdata, disconnect_flags, rc, properties=None):
        print("MQTT: Disconnesso dal broker. I dati verranno bufferizzati.")
        self.connected = False

    def connect(self):
        # Inizializza la connessione con riconnessione automatica
        try:
            self.client.reconnect_delay_set(min_delay=1, max_delay=30)
            self.client.connect(self.broker, self.port, keepalive=60)
            self.client.loop_start()  # Loop asincrono gestito da paho
        except Exception as e:
            print(f"Errore critico MQTT: {e}")

    def _health_check_loop(self):
        # Invia lo stato dell'hardware ogni 5 secondi (QoS 0)
        while not self.stop_health_check.is_set():
            if self.connected:
                health_data = {
                    "cpu_percent": psutil.cpu_percent(interval=None),
                    "memory_percent": psutil.virtual_memory().percent,
                    "process_memory_mb": round(psutil.Process().memory_info().rss / (1024 * 1024), 2),
                    "buffer_usage": len(self.buffer),
                    "t_check": time.time()
                }
                self.send_system_event("system", "health", health_data, qos=0)
            time.sleep(5)

    def send_tracking(self, camera, mode, data, qos=1):
        # Invia dati di tracking (Lat/Lon) con QoS 1
        topic = f"masa/{camera}/{mode}/data"
        self._publish_logic(topic, data, qos)

    def send_system_event(self, camera, event_type, data, qos=2):
        # Invia eventi critici (Summary, Start, Stop) con QoS 2
        topic = f"masa/{camera}/system/{event_type}"
        self._publish_logic(topic, data, qos)

    def _publish_logic(self, topic, data, qos):
        payload = msgpack.packb(data, use_bin_type=True)

        if self.connected:
            self.client.publish(topic, payload, qos=qos)
        elif qos > 0:
            # Salvataggio nel buffer dei messaggi che richiedono consegna garantita
            self.buffer.append((topic, payload, qos))

    def _flush_buffer(self):
        if not self.buffer:
            return

        print(f"Riconnessione: invio di {len(self.buffer)} pacchetti accumulati...")
        while self.buffer and self.connected:
            topic, payload, qos = self.buffer.popleft()
            self.client.publish(topic, payload, qos=qos)
        print("Buffer locale svuotato.")

    def disconnect(self):
        self.send_system_event("system", "status", {"status": "SHUTDOWN"}, qos=2)
        self.stop_health_check.set()
        if self.health_thread:
            self.health_thread.join(timeout=2)
        self.client.loop_stop()
        self.client.disconnect()
        print("MQTT: Client disconnesso.")