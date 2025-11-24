import os
import json
from collections import deque
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class MetricsPlotter:
    def __init__(self, save_plots: bool = False, output_dir: str = "plots"):

        self.save_plots = save_plots
        self.output_dir = output_dir

        if self.save_plots:
            os.makedirs(self.output_dir, exist_ok=True)

    def plot_metric(
            self,
            data: List,
            config: Dict,
            model_name: str,
            tracking_mode: str
    ) -> plt.Figure:

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(
            config['x'],
            config['y'],
            marker=config.get('marker', 'o'),
            linestyle='-',
            color=config['color'],
            label=config['label']
        )

        if config.get('mean'):
            mean_val = np.mean(config['y'])
            ax.axhline(
                mean_val,
                color='red',
                linestyle='--',
                linewidth=2,
                label=f"Media: {mean_val:.2f}{config.get('unit', '')}"
            )

        ax.set_title(f"{config['title']} – Modello: {model_name}", fontsize=14)
        ax.set_xlabel(config['xlabel'], fontsize=12)
        ax.set_ylabel(config['ylabel'], fontsize=12)
        ax.grid(True)
        ax.legend()
        plt.tight_layout()

        # Salva il grafico se richiesto
        if self.save_plots:
            metric_type = config.get('save_name', 'metric')
            filename = os.path.join(
                self.output_dir,
                f"{metric_type}_{model_name}_{tracking_mode}.png"
            )
            fig.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Grafico salvato: {filename}")

        return fig

    def plot_accuracy(
            self,
            track_confidences: Dict,
            model_name: str,
            tracking_mode: str
    ) -> Optional[plt.Figure]:

        if not track_confidences:
            print("Nessun oggetto trovato: Grafico accuracy non generato")
            return None

        track_ids = list(track_confidences.keys())
        confidences = [conf for (_, conf) in track_confidences.values()]
        class_labels = [cls for (cls, _) in track_confidences.values()]

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(track_ids, confidences, color='skyblue')

        # Aggiungi etichette sulle barre
        for tid, conf, cls in zip(track_ids, confidences, class_labels):
            ax.text(
                tid,
                conf + 0.02,
                f"{cls}\n{conf:.2f}",
                ha='center',
                fontsize=9
            )

        mean_conf = np.mean(confidences)
        ax.axhline(
            mean_conf,
            color='red',
            linestyle='--',
            linewidth=2,
            label=f"Media: {mean_conf:.2f}"
        )

        ax.set_title(f"Accuracy YOLO su tracciamento – Modello: {model_name}", fontsize=14)
        ax.set_xlabel("ID Tracciamento", fontsize=12)
        ax.set_ylabel("Accuracy YOLO", fontsize=12)
        ax.set_ylim(0, 1.05)
        ax.grid(True)
        ax.legend()
        plt.tight_layout()

        if self.save_plots:
            filename = os.path.join(
                self.output_dir,
                f"accuracy_{model_name}_{tracking_mode}.png"
            )
            fig.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Grafico salvato: {filename}")

        return fig

    def plot_fps(
            self,
            fps_data: deque,
            timestamps: deque,
            model_name: str,
            tracking_mode: str
    ) -> Optional[plt.Figure]:

        if not fps_data:
            print("Nessun dato FPS: Grafico FPS non generato")
            return None

        fps_values = list(fps_data)
        time_indices = list(timestamps)

        return self.plot_metric(
            fps_values,
            {
                'x': time_indices,
                'y': fps_values,
                'color': 'blue',
                'label': 'FPS per frame',
                'mean': True,
                'title': 'Andamento FPS',
                'xlabel': 'Secondi',
                'ylabel': 'FPS',
                'save_name': 'fps'
            },
            model_name,
            tracking_mode
        )

    def plot_memory(
            self,
            memory_data: deque,
            timestamps: deque,
            model_name: str,
            tracking_mode: str
    ) -> Optional[plt.Figure]:

        if not memory_data:
            print("Nessun dato memoria: Grafico memoria non generato")
            return None

        mem_values = list(memory_data)
        time_indices = list(timestamps)

        return self.plot_metric(
            mem_values,
            {
                'x': time_indices,
                'y': mem_values,
                'marker': 's',
                'color': 'purple',
                'label': 'Memoria (MB)',
                'mean': True,
                'unit': ' MB',
                'title': 'Andamento utilizzo RAM',
                'xlabel': 'Tempo (secondi)',
                'ylabel': 'Memoria utilizzata (MB)',
                'save_name': 'memory'
            },
            model_name,
            tracking_mode
        )

    def generate_all_plots(
            self,
            fps_data: deque,
            memory_data: deque,
            track_confidences: Dict,
            model_path: str,
            timestamps: deque,
            tracking_mode: str
    ) -> List[plt.Figure]:

        model_name = os.path.basename(model_path).replace('.pt', '')
        figures = []

        print("Generazione dei grafici delle performance...")

        # Grafico Accuracy
        fig = self.plot_accuracy(track_confidences, model_name, tracking_mode)
        if fig:
            figures.append(fig)
        # Grafico FPS
        fig = self.plot_fps(fps_data, timestamps, model_name, tracking_mode)
        if fig:
            figures.append(fig)
        # Grafico Memoria
        fig = self.plot_memory(memory_data, timestamps, model_name, tracking_mode)
        if fig:
            figures.append(fig)

        if figures:
            plt.show(block=True)

            for fig in figures:
                plt.close(fig)
        else:
            print("Nessun grafico generato (dati insufficienti)")

        return figures

    def save_metrics_to_json(
            self,
            fps_data: deque,
            memory_data: deque,
            track_confidences: Dict,
            timestamps: deque,
            model_path: str,
            tracking_mode: str,
            filename: Optional[str] = None
    ):

        model_name = os.path.basename(model_path).replace('.pt', '')

        if filename is None:
            filename = os.path.join(
                self.output_dir,
                f"metrics_{model_name}_{tracking_mode}.json"
            )

        metrics = {
            'model': model_name,
            'tracking_mode': tracking_mode,
            'fps': {
                'values': list(fps_data),
                'mean': float(np.mean(list(fps_data))) if fps_data else 0,
                'std': float(np.std(list(fps_data))) if fps_data else 0,
                'min': float(np.min(list(fps_data))) if fps_data else 0,
                'max': float(np.max(list(fps_data))) if fps_data else 0
            },
            'memory': {
                'values': list(memory_data),
                'mean': float(np.mean(list(memory_data))) if memory_data else 0,
                'std': float(np.std(list(memory_data))) if memory_data else 0,
                'min': float(np.min(list(memory_data))) if memory_data else 0,
                'max': float(np.max(list(memory_data))) if memory_data else 0
            },
            'accuracy': {
                'tracks': {
                    str(k): {'class': v[0], 'confidence': v[1]}
                    for k, v in track_confidences.items()
                },
                'mean': float(np.mean([v[1] for v in track_confidences.values()])) if track_confidences else 0
            },
            'timestamps': list(timestamps)
        }

        with open(filename, 'w') as f:
            json.dump(metrics, f, indent=2)

        print(f"Metriche salvate in: {filename}")

def generate_performance_plots(
        fps_data: deque,
        memory_data: deque,
        track_confidences: Dict,
        model_path: str,
        class_names: Dict,
        timestamps: deque,
        tracking_mode: str,
        save_plots: bool = False
):

    plotter = MetricsPlotter(save_plots=save_plots)
    plotter.generate_all_plots(
        fps_data=fps_data,
        memory_data=memory_data,
        track_confidences=track_confidences,
        model_path=model_path,
        timestamps=timestamps,
        tracking_mode=tracking_mode
    )

def load_session_data(session_dir: str) -> Dict:

    metrics_path = os.path.join(session_dir, "metrics.csv")
    tracks_path = os.path.join(session_dir, "tracks.csv")
    config_path = os.path.join(session_dir, "config.json")

    with open(config_path, 'r') as f:
        config = json.load(f)

    metrics_df = pd.read_csv(metrics_path)
    tracks_df = pd.read_csv(tracks_path)

    # Calcola confidence per track (prende il primo valore)
    track_confidences = {}
    for track_id in tracks_df['track_id'].unique():
        track_data = tracks_df[tracks_df['track_id'] == track_id].iloc[0]
        track_confidences[track_id] = (
            track_data['class_name'],
            track_data['confidence']
        )

    return {
        'config': config,
        'metrics': metrics_df,
        'tracks': tracks_df,
        'track_confidences': track_confidences,
        'timestamps': metrics_df['timestamp'].tolist(),
        'fps_data': metrics_df['fps'].tolist(),
        'memory_data': metrics_df['memory_mb'].tolist()
    }

def generate_performance_plots_from_csv(
        session_dir: str,
        plots_dir: str,
        save_plots: bool = True
):

    print(f"\n Caricamento dati da: {session_dir}")

    data = load_session_data(session_dir)

    plotter = MetricsPlotter(save_plots=save_plots, output_dir=plots_dir)

    model_name = os.path.basename(data['config']['yolo_model_path']).replace('.pt', '')
    tracking_mode = data['config']['tracking_mode']

    # Converti liste in deque
    from collections import deque
    fps_data = deque(data['fps_data'])
    memory_data = deque(data['memory_data'])
    timestamps = deque(data['timestamps'])

    # Genera tutti i grafici
    plotter.generate_all_plots(
        fps_data=fps_data,
        memory_data=memory_data,
        track_confidences=data['track_confidences'],
        model_path=data['config']['yolo_model_path'],
        timestamps=timestamps,
        tracking_mode=tracking_mode
    )

    print(f" Grafici salvati in: {plots_dir}")