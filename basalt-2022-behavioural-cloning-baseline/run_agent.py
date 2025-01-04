from argparse import ArgumentParser
import pickle
import pygame
import aicrowd_gym
import minerl
import numpy as np
import cv2

from openai_vpt.agent import MineRLAgent

def main(model, weights, env, n_episodes=1, max_steps=int(1e9), show=True):

    # Imposta a True per avviare una simulazione di 10 episodi da 1500 steps con log dell'inventario
    TEST_x10_1500_STEP = False

    # Using aicrowd_gym is important! Your submission will not work otherwise
    env = aicrowd_gym.make(env)
    agent_parameters = pickle.load(open(model, "rb"))
    policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
    device = "cpu"  # Forza l'uso della CPU
    agent = MineRLAgent(env, device=device, policy_kwargs=policy_kwargs, pi_head_kwargs=pi_head_kwargs)

    agent.load_weights(weights)

    screen = pygame.display.set_mode((1280, 720))
    pygame.display.set_caption('Minecraft agent')

    if TEST_x10_1500_STEP:
        n_episodes = 10
        max_steps = 1500

    for _ in range(n_episodes):

        obs = env.reset()
        #env.seed(7011)

        for _ in range(max_steps):
            # Gestione degli eventi per controllare l'input utente
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    pygame.quit()
                    exit()
                if event.type == pygame.KEYDOWN and event.key == pygame.K_q:  # Controlla se il tasto Q è premuto
                    env.close()
                    pygame.quit()
                    exit()

            action = agent.get_action(obs)

            # ESC is not part of the predictions model.
            # For baselines, we just set it to zero.
            # We leave proper execution as an exercise for the participants :)
            action["ESC"] = 0

            obs, _, done, _ = env.step(action)
            if show:
                image = obs["pov"]  # Ottieni l'immagine dall'osservazione

                # Modifiche all'immagine per farla piacere a pygame
                image = np.flip(image, axis=1)  # Specchia orizzontalmente
                image = np.rot90(image)  # Ruota di 90°
                image = cv2.resize(image, (720, 1280))  # Cambia la risoluzione
                image = image.astype(np.uint8)

                # Rendering
                surface = pygame.surfarray.make_surface(image)
                screen.blit(surface, (0, 0))
                pygame.display.update()
            if done:
                break

        if TEST_x10_1500_STEP:
            inventory = obs["inventory"]
            available_items = {item: int(quantity) for item, quantity in inventory.items() if quantity > 0}

            print("------------------- esito -------------------")
            print("Oggetti disponibili nell'inventario:")
            for item, quantity in available_items.items():
                print(f"{item}: {quantity}")
            print("------------------- esito -------------------")

    env.close()

if __name__ == "__main__":
    parser = ArgumentParser("Run pretrained models on MineRL environment")

    parser.add_argument("--weights", type=str, required=True, help="Path to the '.weights' file to be loaded.")
    parser.add_argument("--model", type=str, required=True, help="Path to the '.model' file to be loaded.")
    parser.add_argument("--env", type=str, required=True)
    parser.add_argument("--show", action="store_true", help="Render the environment.")

    args = parser.parse_args()

    main(args.model, args.weights, args.env, show=args.show)
