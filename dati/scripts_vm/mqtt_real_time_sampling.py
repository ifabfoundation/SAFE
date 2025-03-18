import paho.mqtt.client as mqtt
import time
import csv
import os
from collections import defaultdict

# Variabili per memorizzare i dati temporanei
dati_raccolti = defaultdict(list)

# Callback quando il client si connette al broker
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connesso al broker MQTT con successo")
        # Sottoscriviti a tutti i topic
        client.subscribe("#")  # "#" si sottoscrive a tutti i topic
    else:
        print("Connessione al broker fallita, codice di risultato: ", rc)

# Callback quando arriva un messaggio
def on_message(client, userdata, msg):
    # Decodifica il messaggio ricevuto
    try:
        messaggio = float(msg.payload.decode())  # Converti il messaggio in float, presupponendo che sia numerico
        dati_raccolti[msg.topic].append(messaggio)  # Salva il valore per quel topic
        salva_valore(msg.topic, messaggio)
    except ValueError:
        print(f"Messaggio non numerico ricevuto su {msg.topic}: {msg.payload.decode()}")

# Funzione per salvare i valori ricevuti nel CSV
def salva_valore(topic, valore):
    file_path = "/home/ifabadmin/data_streaming/dati_mqtt_realtime.csv"
    with open(file_path, "a", newline='') as csvfile:
        writer = csv.writer(csvfile)
        timestamp = time.ctime()
        # Scrivi il valore nel CSV
        writer.writerow([topic, valore, timestamp])
        print(f"Salvato valore: {topic}, {valore}, {timestamp}")

# Crea il file CSV e scrivi l'intestazione (solo la prima volta)
file_path = "/home/ifabadmin/data_streaming/dati_mqtt_realtime.csv"
if not os.path.exists(file_path):
    with open(file_path, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Topic", "Valore", "Timestamp"])

# Configura il client MQTT
client = mqtt.Client()

# Assegna i callback
client.on_connect = on_connect
client.on_message = on_message

# Connessione al broker Mosquitto (localhost se è sullo stesso server)
broker_address = "localhost"  # Cambia questo indirizzo se il broker è remoto
client.connect(broker_address, 1883, 60)

# Mantieni la connessione attiva e processa i messaggi
client.loop_forever()

