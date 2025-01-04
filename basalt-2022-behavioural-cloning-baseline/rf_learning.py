from argparse import ArgumentParser
import pickle
import os
import aicrowd_gym
import minerl
import torch as th
import numpy as np
from collections import deque
import matplotlib
matplotlib.use("TkAgg")  # Usa il backend interattivo TkAgg
import matplotlib.pyplot as plt
import pandas as pd

from openai_vpt.agent import MineRLAgent

# Nome del file Excel
EXCEL_FILE = "reward_log.xlsx"

# Funzione per calcolare la reward in base ai materiali
MATERIAL_REWARDS = {
    "birch_log": 1.2,
    "dark_oak_log": 1.2,
    "jungle_log": 1.2,
    "oak_log": 1.2,
    "spruce_log": 1.2,
    "dark_oak_planks": 1.5,
    "jungle_planks": 1.5,
    "oak_planks": 1.5,
    "spruce_planks": 1.5,
    "crafting_table": 2.0,  # Ricompensa per creare una crafting table
    "dirt": -0.2,
    "gravel": -0.5,
    "sand": -0.3
}

# Configurazione per monitorare i salti
JUMP_THRESHOLD = 10
JUMP_WINDOW = 20  # Numero massimo di passi da considerare

# Normalizzazione della reward
def compute_reward(inventory, prev_inventory):
    reward = 0
    for material, value in MATERIAL_REWARDS.items():
        current_quantity = inventory.get(material, 0)
        previous_quantity = prev_inventory.get(material, 0)
        if current_quantity > previous_quantity:
            reward += (current_quantity - previous_quantity) * value
    # Normalizza la reward su un intervallo standardizzato
    return reward / (abs(reward) + 1e-6) if abs(reward) > 1 else reward

# Funzione per assegnare reward basate su azioni
LOG_TYPES = ["birch_log", "dark_oak_log", "jungle_log", "oak_log", "spruce_log"]
def action_based_reward(action, inventory, jump_window, inventory_reward_given):
    """
    Calcola la reward basata sulle azioni dell'agente.
    
    Args:
        action: L'azione eseguita dall'agente.
        inventory: Lo stato corrente dell'inventario dell'agente.
        jump_window: Una deque che traccia i salti negli ultimi N passi.
        isInventoryOpen: Booleano che indica se l'inventario è aperto.
        
    Returns:
        La reward associata all'azione.
    """
    reward = 0
    # Reward positiva se l'agente apre l'inventario e ha un log ma non una crafting table
    if (any(inventory.get(log, 0) > 0 for log in LOG_TYPES) and
        inventory.get("crafting_table", 0) == 0 and
        action.get("inventory", 0) == 1 and not inventory_reward_given):
        reward += 0.5
        inventory_reward_given = True  # La reward è stata assegnata
    
    # Reward negativa se in una finestra di 20 passi ci sono più di 10 salti
    if sum(jump_window) > 10:  # Usa la finestra mobile per monitorare i salti
        reward -= 0.18
       
    # Penalità per inattività: nessuna azione con valore 1
    if not any(np.any(value == 1) if isinstance(value, np.ndarray) else value == 1 for value in action.values()):
        reward -= 0.1

    return reward / (abs(reward) + 1e-6) if abs(reward) > 1 else reward, inventory_reward_given

    
# Funzione per normalizzare la perdita
def normalize(tensor):
    if tensor.numel() <= 1:  # Se il tensore ha un solo elemento o è vuoto
        return tensor  # Restituisci il tensore originale senza normalizzare
    return (tensor - tensor.mean()) / (tensor.std() + 1e-8)

def log_to_excel(step, cumulative_reward):
    # Controlla se il file esiste
    if os.path.exists(EXCEL_FILE):
        df = pd.read_excel(EXCEL_FILE)
    else:
        # Crea un nuovo DataFrame se il file non esiste
        df = pd.DataFrame(columns=["Step", "Cumulative Reward"])

    # Crea un DataFrame per i nuovi dati
    new_data = pd.DataFrame({"Step": [step], "Cumulative Reward": [cumulative_reward]})
    
    # Usa pd.concat per aggiungere i nuovi dati
    df = pd.concat([df, new_data], ignore_index=True)
    
    # Scrivi i dati nel file Excel
    df.to_excel(EXCEL_FILE, index=False)



def main(model, weights, env, n_episodes=3, max_steps=int(1e9), show=True):
    env = aicrowd_gym.make(env)
    agent_parameters = pickle.load(open(model, "rb"))
    policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
    agent = MineRLAgent(env, policy_kwargs=policy_kwargs, pi_head_kwargs=pi_head_kwargs)
    agent.load_weights(weights)

    # Variabili per monitorare la reward cumulativa
    cumulative_rewards = []  # Salva la reward cumulativa per ogni episodio

    plt.ion()  # Abilita il grafico interattivo
    fig, ax = plt.subplots()
    ax.set_title("Reward Cumulativa in Tempo Reale")
    ax.set_xlabel("Passo")
    ax.set_ylabel("Reward Cumulativa")
    line, = ax.plot([], [], label="Reward Cumulativa")
    plt.legend()
    plt.grid(True)

    for episode in range(n_episodes):
        obs = env.reset()
        prev_inventory = {key: 0 for key in MATERIAL_REWARDS.keys()}
        
        cumulative_episode_reward = 0
        episode_rewards = []  # Salva le reward per ogni passo
        steps = []  # Salva i passi per il grafico

        # Crea una cartella per l'episodio
        episode_dir = f"./data/Stats/RLTranding_reward/episode{episode + 1}"
        os.makedirs(episode_dir, exist_ok=True)

        for step in range(max_steps):
            action = agent.get_action(obs)
            action["ESC"] = 0
            obs, _, done, _ = env.step(action)
            
            # Calcola la reward basata sull'inventario
            inventory = obs["inventory"]
            reward = compute_reward(inventory, prev_inventory)
            cumulative_episode_reward += reward
            
            # Aggiorna i dati per il grafico
            steps.append(step + 1)
            episode_rewards.append(cumulative_episode_reward)

            # **Aggiorna i dati del grafico senza rigenerare la finestra**
            line.set_xdata(steps)
            line.set_ydata(episode_rewards)
            ax.relim()  # Ricalcola i limiti dell'asse
            ax.autoscale_view()  # Adatta la vista agli aggiornamenti
            fig.canvas.draw_idle()  # Aggiorna il canvas in modo efficiente
            plt.pause(0.01)  # Introduce una breve pausa per permettere il rendering

            # Aggiorna lo stato dell'inventario
            prev_inventory = {key: inventory.get(key, 0) for key in MATERIAL_REWARDS.keys()}

            if show:
                env.render()
            if done:
                break

        # Salva dati e grafico per l'episodio
        df = pd.DataFrame({"Passo": steps, "Reward Cumulativa": episode_rewards})
        episode_excel_file = os.path.join(episode_dir, f"episode_{episode + 1}_rewards.xlsx")
        df.to_excel(episode_excel_file, index=False)

        episode_graph_file = os.path.join(episode_dir, f"episode_{episode + 1}_reward_graph.png")
        plt.figure()
        plt.plot(steps, episode_rewards, marker='o', label=f"Episode {episode + 1}")
        plt.title(f"Andamento Reward Cumulativa - Episodio {episode + 1}")
        plt.xlabel("Passo")
        plt.ylabel("Reward Cumulativa")
        plt.grid(True)
        plt.legend()
        plt.savefig(episode_graph_file)
        plt.close()

        print(f"Episodio {episode + 1}/{n_episodes} - Reward cumulativa: {cumulative_episode_reward}")
        cumulative_rewards.append(cumulative_episode_reward)

    env.close()

    # Genera il grafico complessivo per tutti gli episodi
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, n_episodes + 1), cumulative_rewards, marker='o', label="Reward Cumulativa Totale")
    plt.title("Andamento della Reward Cumulativa per Tutti gli Episodi", fontsize=14)
    plt.xlabel("Episodio", fontsize=12)
    plt.ylabel("Reward Cumulativa", fontsize=12)
    plt.grid(True)
    plt.legend()
    plt.savefig("./data/Stats/RLTranding_reward/cumulative_reward_trend_all_episodes.png")
    plt.show()


if __name__ == "__main__":
    parser = ArgumentParser("PPO Reinforcement Learning Execution with RMSProp")

    parser.add_argument("--env", type=str, required=True, help="Nome dell'ambiente MineRL (es. MineRLObtainDiamondShovel-v0)")
    parser.add_argument("--model", type=str, required=True, help="Percorso al file '.model' pre-addestrato.")
    parser.add_argument("--weights", type=str, required=True, help="Percorso al file '.weights' pre-addestrato.")
    parser.add_argument("--episodes", type=int, default=10, help="Numero di episodi da eseguire.")
    parser.add_argument("--max-steps", type=int, default=2000, help="Numero massimo di passi per episodio.")
    parser.add_argument("--show", action="store_true", help="Visualizza il rendering dell'ambiente.")

    args = parser.parse_args()

    main(
        model=args.model,
        weights=args.weights,
        env=args.env,
        n_episodes=args.episodes,
        max_steps=args.max_steps,
        show=args.show
    )
