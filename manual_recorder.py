import minerl
import gym
import cv2
from pynput import keyboard, mouse

# Crea l'ambiente
env = gym.make("MineRLBasaltFindCave-v0")
obs = env.reset()

# Configura il writer video
height, width, _ = obs["pov"].shape
video = cv2.VideoWriter("manual_episode.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (width, height))

# Variabili per gestire le azioni
current_action = env.action_space.noop()  # Azione base nulla
stop_program = False  # Flag per terminare il programma

# Variabili per tracciare la posizione del cursore
previous_x, previous_y = width // 2, height // 2  # Centro dello schermo
mouse_sensitivity = 0.2  # Sensibilità del mouse


def on_press(key):
    global stop_program
    try:
        if hasattr(key, "char") and key.char:
            if key.char == 'w':  # Cammina avanti
                current_action["forward"] = 1
            elif key.char == 'a':  # Vai a sinistra
                current_action["left"] = 1
            elif key.char == 'd':  # Vai a destra
                current_action["right"] = 1
            elif key.char == 's':  # Vai indietro
                current_action["back"] = 1
            elif key.char == 'e':  # Apri l'inventario
                current_action["inventory"] = 1
        # Controllo esplicito per il tasto Spazio
        if key == keyboard.Key.space:
            current_action["jump"] = 1
        # Controllo esplicito per il tasto ESC
        if key == keyboard.Key.esc:
            stop_program = True
    except AttributeError:
        pass


def on_release(key):
    try:
        if hasattr(key, "char") and key.char:
            if key.char == 'w':
                current_action["forward"] = 0
            elif key.char == 'a':
                current_action["left"] = 0
            elif key.char == 'd':
                current_action["right"] = 0
            elif key.char == 's':
                current_action["back"] = 0
            elif key.char == 'e':
                current_action["inventory"] = 0
        # Controllo esplicito per il tasto Spazio
        if key == keyboard.Key.space:
            current_action["jump"] = 0
    except AttributeError:
        pass


def on_click(x, y, button, pressed):
    if button == mouse.Button.left:
        current_action["attack"] = 1 if pressed else 0
    elif button == mouse.Button.right:
        current_action["use"] = 1 if pressed else 0


def on_move(x, y):
    global previous_x, previous_y, current_action

    # Calcola i delta del mouse rispetto alla posizione precedente
    dx = (x - previous_x) * mouse_sensitivity
    dy = (y - previous_y) * mouse_sensitivity

    # Se c'è movimento, aggiorna la visuale
    if dx != 0 or dy != 0:
        current_action["camera"] = [dy, dx]
        # Aggiorna la posizione precedente
        previous_x, previous_y = x, y
    else:
        # Se non c'è movimento, azzera la camera
        current_action["camera"] = [0.0, 0.0]


# Listener per tastiera e mouse
keyboard_listener = keyboard.Listener(on_press=on_press, on_release=on_release)
mouse_listener = mouse.Listener(on_click=on_click, on_move=on_move)

# Avvia i listener
keyboard_listener.start()
mouse_listener.start()

# Loop principale
try:
    while not stop_program:
        env.render()

        # Esegui l'azione attuale
        obs, reward, done, info = env.step(current_action)

        # Salva il frame nel video
        video.write(cv2.cvtColor(obs["pov"], cv2.COLOR_RGB2BGR))

        # Interruzione del ciclo principale se l'ambiente è completato
        if done:
            print("Ambiente completato.")
            break

finally:
    env.close()
    video.release()
    print("Video salvato in 'manual_episode.mp4'")

    # Ferma i listener
    keyboard_listener.stop()
    mouse_listener.stop()
