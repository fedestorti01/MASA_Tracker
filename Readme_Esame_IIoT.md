# Smart Traffic Control System

Questo progetto implementa un sistema di controllo del traffico intelligente. Il sistema utilizza il protocollo **MQTT** e la serializzazione **Msgpack** per monitorare la densità del traffico in tempo reale e adattare i cicli semaforici per ottimizzare il flusso stradale.

## ️ Architettura del Sistema

Il progetto è diviso in tre componenti principali che comunicano in modo asincrono:

1.  **Synthetic Traffic Camera (`synthetic_traffic_camera.py`)**: 
    * Simula una telecamera intelligente posizionata in un'area MASA.
    * Genera dati stocastici simulando fasi di **Congestione** (affollamento) e di traffico **Regolare**.
    * Invia messaggi Msgpack contenenti: `track_id`, `class` (veicolo), `lat`, `long` e `timestamp`.

2.  **Traffic Data Collector (`traffic_data_collector.py`)**:
    * Sottoscrive i dati della camera.
    * Utilizza una **Sliding Window di 10 secondi** per analizzare il traffico.
    * Applica una logica di **conteggio univoco** (tramite ID veicolo).
    * Soglia di attivazione: **≥ 6 veicoli univoci** per lo stato `PRIORITY`.

3.  **Traffic Light Visualizer (`traffic_light_visualizer.py`)**:
    * Interfaccia grafica che rappresenta l'attuatore fisico.
    * Riceve i comandi (`NORMAL` o `PRIORITY`) e gestisce il ciclo semaforico.
    * **Sicurezza garantita**: Anche in modalità priorità, il semaforo rispetta sempre la sequenza Verde-Giallo-Rosso.

---
## Requisiti Tecnici

* **Sistema Operativo**: Windows.
* **Broker MQTT**: Mosquitto (attivo su `localhost:1883`).
* **Librerie Python**:
    * `paho-mqtt`
    * `msgpack`
    * `tkinter` (per interfaccia grafica semaforo)
---

##  Logica Operativa

### Fasi della Camera (Simulazione Stocastica)
* **Fase Regolata**: Generazione veicoli sporadica (Probabilità 15% ogni 1.0s). Durata: 20-30 secondi.
* **Fase di Congestione**: Generazione intensa (Probabilità 80% ogni 0.7s). Durata: 10-15 secondi.

### Ciclo Semaforico (Visualizer)
* **Modalità NORMAL**: 
    * Verde (5s)  → Giallo (2s) → Rosso (5s) 
* **Modalità PRIORITY**: 
    * Se attivo, il semaforo mantiene il **Verde fisso** per permettere il deflusso.
    * Se il comando arriva durante il Giallo o Rosso, il semaforo completa il tempo di sicurezza e poi scatta in Verde prioritario.
---

## Come avviare il progetto

1.  **Avviare il Broker MQTT**:
    Assicurarsi che il servizio `mosquitto` sia attivo sul PC.

2.  **Avviare i componenti** (in tre terminali separati):
    ```bash
    python traffic_light_visualization.py
    python traffic_data_collector.py
    python synthetic_traffic_camera.py
    ```

## Formato Dati (MQTT Payload)
Il sistema scambia dati binari tramite Msgpack per massimizzare l'efficienza:
```json
{
    "track_id": 1,
    "class": "car",
    "lat": 44.648xxx,
    "long": 10.920xxx,
    "timestamp": 1771085940.0
    }