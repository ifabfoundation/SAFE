import paho.mqtt.client as mqtt
import time
import csv
import os
from collections import defaultdict

dati_raccolti = defaultdict(list)
ultimo_salvataggio = time.time()

def on_connect(client, userdata, flags, rc):
	if rc == 0:
		print("Connesso al broker MQTT con successo")
		client.subscribe("#")
	else:
		print("Connessione al broker fallita, codice risultato: ", rc)

def on_message(client, userdata, msg):
	try:
		messaggio = float(msg.payload.decode())
		dati_raccolti[msg.topic].append(messaggio)
	except ValueError:
		print(f"Messaggio non numerico ricevuto su {msg.topic}: {msg.payload.decode()}")
	global ultimo_salvataggio
	if time.time() - ultimo_salvataggio >= 60:
		salva_campionamento()
		ultimo_salvataggio = time.time()

def salva_campionamento():
	with open ("dati_mqtt_campionati.csv", "a", newline='') as csvfile:
		writer = csv.writer(csvfile)
		for topic, valori in dati_raccolti.items():
			if len(valori) > 0:
				media = sum(valori) / len(valori)
			else:
				media = None
			timestamp = time.ctime()
			writer.writerow([topic, media, timestamp])
	dati_raccolti.clear()


if not os.path.exists("dati_mqtt_campionati.csv"):
	with open("dati_mqtt_campionati.csv", "w", newline='') as csvfile:
		writer = csv.writer(csvfile)
		writer.writerow(["Topic", "Media_Valori", "Timestamp"])

client = mqtt.Client()

client.on_connect = on_connect
client.on_message = on_message

broker_address =  "localhost"
client.connect(broker_address, 1883, 60)

client.loop_forever()

