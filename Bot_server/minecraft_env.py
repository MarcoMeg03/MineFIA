import minerl
import gym
import cv2
import numpy as np
import csv
import time
import os
import json
import websocket
import coloredlogs
import logging

coloredlogs.install(logging.DEBUG)

# Configura connessione WebSocket
ws = websocket.create_connection("ws://localhost:8080")

def send_command(action, duration=1000):
    command = json.dumps({"action": action, "duration": duration})
    ws.send(command)

# Funzione di agente semplice, aggiornata per compatibilità con `minerl`
def simple_agent(obs, step_count):
    action = env.action_space.no_op()

    pov = obs.get('pov', None)
    if pov is None:
        return action  # Evita errori se 'pov' è assente

    green_pixels = np.sum(pov[:, :, 1] > 150)

    if green_pixels > 1000:
        send_command('forward', 1000)
        send_command('jump', 500)
        send_command('attack', 500)

    return action

# Crea l'ambiente MineRL
env = gym.make('MineRLBasaltFindCave-v0')
obs = env.reset()
episode_count = 0

# Configura il salvataggio del video
height, width, _ = obs['pov'].shape
fps = 20
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_writer = cv2.VideoWriter('minerl_video.avi', fourcc, fps, (width, height))

# Ciclo principale per eseguire le azioni
while episode_count < 2:
    obs = env.reset()
    done = False
    step = 0

    while not done:
        action = simple_agent(obs, step)

        if action is not None:
            obs, reward, done, info = env.step(action)

            if 'error' in info:
                print("Errore nel passo, termina episodio.")
                done = True
                break

            # Salva il frame della POV nell'oggetto VideoWriter
            img = obs['pov']
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # Converti da RGB a BGR per OpenCV
            video_writer.write(img)

            # Visualizza il frame in una finestra (opzionale)
            cv2.imshow('MineRL POV', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                done = True
                break

        step += 1

    episode_count += 1

# Chiudi risorse
env.close()
video_writer.release()
cv2.destroyAllWindows()
ws.close()
print("Ambiente chiuso e connessione terminata.")
