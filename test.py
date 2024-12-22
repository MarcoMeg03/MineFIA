import minerl
import gym
import cv2
import numpy as np
import csv
import time
import coloredlogs
import logging
import os

# Configura il logging
coloredlogs.install(logging.DEBUG)

# Parametri del server remoto
SERVER_IP = "172.20.16.1"
SERVER_PORT = 10000  # Porta standard di Minecraft Malmo

# Modifica variabile d'ambiente di Malmo per puntare al server remoto
os.environ["MALMO_MINECRAFT_IP"] = SERVER_IP

def log_data(step, action, reward, done):
    """Log dei dati su file CSV."""
    with open('minerl_log.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([step, action, reward, done])

# Funzione di un agente semplice basato su regole
def simple_agent(obs, step_count):
    action = env.action_space.noop()
    
    # Estrai l'osservazione del POV
    pov = obs['pov']
    
    # Conta i pixel verdi nell'immagine (ad esempio erba)
    green_pixels = np.sum(pov[:, :, 1] > 150)  # Pixel verdi per l'erba

    # Se trova tanto verde, cammina avanti, ma ogni tanto scava
    if green_pixels > 1000:
        action['forward'] = 1
        action['jump'] = 1  # Salta per evitare ostacoli
        if step_count % 10 == 0:  # Ogni 10 passi, prova a scavare
            action['attack'] = 1
    else:
        # Se non vede tanto verde, cambia direzione e guarda in giro
        action['jump'] = 1  # Salta per evitare ostacoli
        action['camera'] = [0, np.random.uniform(-5, 5)]  # Guarda intorno in modo casuale
        if step_count % 5 == 0:
            action['back'] = 1  # A volte cammina indietro per esplorare

    # Aggiungi azione di scavare se vede blocchi davanti
    if np.mean(pov) < 80:  # Se la vista è scura, potrebbe esserci una parete
        action['attack'] = 1  # Inizia a scavare

    return action

# Crea l'ambiente
env = gym.make('MineRLBasaltFindCave-v0')

# Imposta il numero di frame al secondo
fps = 45

# Ottieni le dimensioni dell'ambiente per il video
obs = env.reset()
height, width, _ = obs['pov'].shape

# Inizializza il video writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_writer = cv2.VideoWriter('minerl_video.avi', fourcc, fps, (width, height))

# Inizializza il numero totale di episodi
total_episodes = 2
episode_count = 0

# Crea il file CSV per il logging
with open('minerl_log.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Step', 'Action', 'Reward', 'Done'])

# Inizia il ciclo principale per gli episodi
while episode_count < total_episodes:
    print(f"Inizio episodio {episode_count + 1}.")
    obs = env.reset()  # Reset dell'ambiente all'inizio di ogni episodio
    step = 0
    done = False
    episode_reward = 0

    while not done:
        action = simple_agent(obs, step)
        obs, reward, done, info = env.step(action)
        episode_reward += reward

        img = obs['pov']
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        video_writer.write(img)
        cv2.imshow('MineRL', img)

        log_data(step, action, reward, done)

        key = cv2.waitKey(int(1000 / fps)) & 0xFF
        if key == ord('q'):
            print("Uscita dal programma.")
            done = True
            break
        elif key == ord('w'):
            print("Episodio terminato con 'w'.")
            done = True
            break

        step += 1
        print(f"Step: {step}, Reward: {reward}, Done: {done}")

    print(f"Episodio {episode_count + 1} terminato con ricompensa totale: {episode_reward}")
    episode_count += 1

env.close()
video_writer.release()
cv2.destroyAllWindows()
print("Ambiente chiuso e video salvato.")
