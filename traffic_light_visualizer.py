import tkinter as tk
import paho.mqtt.client as mqtt
import threading
import time

class TrafficLightGUI:
    def __init__(self, root):
        self.root = root
        self.tl_id = "Traffic Light 01"
        self.command_topic = f"masa/infrastructure/trafficlight/{self.tl_id}/command"
        self.mode = "NORMAL"

        self.root.title(f"Visualizzatore Semaforo - {self.tl_id}")
        self.canvas = tk.Canvas(root, width=200, height=500, bg='#333')
        self.canvas.pack(pady=20)

        self.red_light = self.canvas.create_oval(50, 50, 150, 150, fill="#1a0000")
        self.yellow_light = self.canvas.create_oval(50, 200, 150, 300, fill="#1a1a00")
        self.green_light = self.canvas.create_oval(50, 350, 150, 450, fill="#001a00")

        # MQTT Setup
        self.client = mqtt.Client(client_id=f"GUI_{self.tl_id}")
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message

        # Tentativo di connessione
        try:
            self.client.connect("localhost", 1883)
        except:
            print("Errore: Broker non trovato. Avvia Mosquitto!")

        # Thread separati per non bloccare Tkinter
        threading.Thread(target=self.client.loop_forever, daemon=True).start()
        threading.Thread(target=self.traffic_cycle, daemon=True).start()

    def on_connect(self, client, userdata, flags, rc):
        print(f"Semaforo {self.tl_id} pronto. Ascolto su: {self.command_topic}")
        client.subscribe(self.command_topic)

    def on_message(self, client, userdata, msg):
        self.mode = msg.payload.decode()
        print(f"Cambio Modalità Ricevuto: {self.mode}")

    def set_lights(self, r, y, g):
        self.canvas.itemconfig(self.red_light, fill="red" if r else "#1a0000")
        self.canvas.itemconfig(self.yellow_light, fill="yellow" if y else "#1a1a00")
        self.canvas.itemconfig(self.green_light, fill="green" if g else "#001a00")

    def traffic_cycle(self):
        while True:
            self.set_lights(False, False, True)

            if self.mode == "PRIORITY":
                while self.mode == "PRIORITY":
                    time.sleep(1)
                time.sleep(2)
            else:
                time.sleep(5)

            self.set_lights(False, True, False)
            time.sleep(2)

            self.set_lights(True, False, False)
            time.sleep(5)

if __name__ == "__main__":
    root = tk.Tk()
    gui = TrafficLightGUI(root)
    root.mainloop()