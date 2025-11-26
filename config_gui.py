import tkinter as tk
from tkinter import ttk, messagebox
from dataclasses import dataclass
from typing import Optional

@dataclass
class GUIConfig:
    tracking_mode: str
    duration: int

class SimpleConfigGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Configurazione Tracking")
        self.root.geometry("450x200")
        self.root.resizable(False, False)

        self._center_window()

        self.result: Optional[GUIConfig] = None

        self.create_widgets()

        self.root.protocol("WM_DELETE_WINDOW", self.on_cancel)

        self.root.lift()
        self.root.focus_force()

    def _center_window(self):
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')

    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        ttk.Label(
            main_frame,
            text="Algoritmo di Tracking:",
            font=('Arial', 10, 'bold')
        ).grid(row=0, column=0, sticky=tk.W, pady=(0, 5))

        self.algorithm_var = tk.StringVar(value="deepsort")
        self.algorithm_combo = ttk.Combobox(
            main_frame,
            textvariable=self.algorithm_var,
            values=["deepsort", "kalman", "bytetrack", "botsort"],
            state='readonly',
            width=30
        )
        self.algorithm_combo.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 15))

        ttk.Label(
            main_frame,
            text="Durata (secondi, 0 = infinito):",
            font=('Arial', 10, 'bold')
        ).grid(row=2, column=0, sticky=tk.W, pady=(0, 5))

        self.duration_var = tk.StringVar(value="30")
        self.duration_entry = ttk.Entry(
            main_frame,
            textvariable=self.duration_var,
            width=32
        )
        self.duration_entry.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(0, 20))

        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=4, column=0, sticky=(tk.E))

        self.cancel_button = tk.Button(
            button_frame,
            text="Annulla",
            command=self.on_cancel,
            bg='#e0e0e0',
            fg='black',
            width=12,
            height=1,
            font=('Arial', 10)
        )
        self.cancel_button.pack(side=tk.LEFT, padx=(0, 10))

        self.submit_button = tk.Button(
            button_frame,
            text="Avvia",
            command=self.on_submit,
            bg='#4CAF50',
            fg='white',
            width=12,
            height=1,
            font=('Arial', 10, 'bold')
        )
        self.submit_button.pack(side=tk.LEFT)

        main_frame.columnconfigure(0, weight=1)

    ## Funzione pulsante avvia
    def on_submit(self):
        tracking_mode = self.algorithm_var.get()

        try:
            duration = int(self.duration_var.get())

        except ValueError or duration < 0:
            messagebox.showerror(
                "Errore",
                "La durata deve essere un numero intero e positivo."
            )
            return

        # Crea configurazione e chiudi finestra
        self.result = GUIConfig(
            tracking_mode=tracking_mode,
            duration=duration
        )
        self.root.quit()
        self.root.destroy()

    ## Funzione annulla
    def on_cancel(self):
        self.result = None
        self.root.quit()
        self.root.destroy()

    def run(self) -> Optional[GUIConfig]:
        self.root.mainloop()
        return self.result

if __name__ == "__main__":
    gui = SimpleConfigGUI()
    config = gui.run()

    if config:
        print(f"Configurazione ricevuta:")
        print(f"  Tracking Mode: {config.tracking_mode}, Duration: {config.duration}")
    else:
        print("Configurazione annullata")