import paho.mqtt.client as mqtt
import msgpack
import time
from collections import deque


class SmartTrafficLight:
    def __init__(self):
        # Parametri Logica Traffico
        self.history = deque()
        self.time_window = 20
        self.threshold = 6
        self.mode = "NORMAL"  # "NORMAL" o "PRIORITY"

        # Parametri Ciclo Semaforico
        self.states = ["VERDE", "GIALLO", "ROSSO"]
        self.current_state_idx = 0  # Parte da VERDE
        self.durations = {"VERDE": 10, "GIALLO": 4, "ROSSO": 10}
        self.last_state_change = time.time()

    def update_cycle(self):
        now = time.time()
        elapsed = now - self.last_state_change
        current_color = self.states[self.current_state_idx]

        # CONDIZIONE PRIORITY: Forza il verde se c'è traffico
        if self.mode == "PRIORITY":
            if current_color != "VERDE":
                self._change_state(0)  # Salta subito a VERDE
            return  # Blocca il timer finché siamo in PRIORITY

        # CONDIZIONE NORMAL: Ciclo standard a tempo
        if elapsed > self.durations[current_color]:
            next_idx = (self.current_state_idx + 1) % len(self.states)
            self._change_state(next_idx)

    def _change_state(self, next_idx):
        self.current_state_idx = next_idx
        self.last_state_change = time.time()
        print(f"\n>>> [SEMAFORO]: {self.states[self.current_state_idx]} <<<\n")

    def process_traffic(self):
        now = time.time()
        while self.history and (now - self.history[0][0] > self.time_window):
            self.history.popleft()

        unique_count = len(set(v_id for ts, v_id in self.history))

        # Cambio Modalità
        if unique_count >= self.threshold:
            if self.mode != "PRIORITY":
                print(f"--- RILEVATA ALTA AFFLUENZA ({unique_count} veicoli) ---")
                self.mode = "PRIORITY"
        else:
            if self.mode != "NORMAL":
                print(f"--- TRAFFICO DIMINUITO. Ritorno al ciclo standard ---")
                self.mode = "NORMAL"
        return unique_count


def on_message(client, userdata, msg):
    try:
        data = msgpack.unpackb(msg.payload)
        v_id = data.get('id')
        classe = data.get('cls')

        # Stampa come richiesto
        print(
            f"DATO RICEVUTO | ID:{v_id} | Classe:{classe} | Coord: {data.get('lat')} | {data.get('lon')} | Ora: {data.get('t_detection')}")

        if classe in ['car', 'motorcycle']:
            controller.history.append((time.time(), v_id))
    except Exception as e:
        print(f"Errore: {e}")


# Inizializzazione
controller = SmartTrafficLight()
client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
client.on_message = on_message

client.connect("localhost", 1883)
client.subscribe("masa/#")

print("SISTEMA ATTIVO. Controllo semaforico in corso...")

# Loop personalizzato per gestire sia MQTT che il Timer del semaforo
while True:
    client.loop(timeout=0.1)  # Gestisce messaggi MQTT
    controller.process_traffic()  # Analizza densità traffico
    controller.update_cycle()  # Gestisce i colori (Verde/Giallo/Rosso)
