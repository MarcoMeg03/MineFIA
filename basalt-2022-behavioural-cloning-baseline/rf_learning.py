from argparse import ArgumentParser
import pickle
import os
import matplotlib.pyplot as plt
import pandas as pd

import aicrowd_gym
import minerl
import torch as th
import numpy as np
from collections import deque

from openai_vpt.agent import MineRLAgent

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
    "crafting_table": 2.0, 
    "dirt": -0.2,
    "gravel": -0.5,
    "sand": -0.3
}

# Configurazione per monitorare i salti
JUMP_THRESHOLD = 10
# Numero massimo di passi da considerare
JUMP_WINDOW = 20 

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

def main(model, weights, env, n_episodes=3, max_steps=int(1e9), show=True):
    env = aicrowd_gym.make(env)
    agent_parameters = pickle.load(open(model, "rb"))
    policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
    agent = MineRLAgent(env, policy_kwargs=policy_kwargs, pi_head_kwargs=pi_head_kwargs)
    agent.load_weights(weights)

    # Configura il grafico interattivo
    plt.ion()
    fig, ax = plt.subplots()
    ax.set_title("Reward Cumulativa in Tempo Reale")
    ax.set_xlabel("Passi")
    ax.set_ylabel("Reward Cumulativa")
    line, = ax.plot([], [], label="Reward Cumulativa")
    plt.legend()
    plt.grid(True)

    # Congela tutti i parametri del modello
    for param in agent.policy.parameters():
        param.requires_grad = False
        
    # Sblocca i parametri del `pi_head` (responsabili della distribuzione delle azioni)
    for param in agent.policy.pi_head.parameters():
        param.requires_grad = True

    optimizer = th.optim.RMSprop(
        filter(lambda p: p.requires_grad, agent.policy.parameters()), 
        lr=0.00001, 
        alpha=0.99, 
        eps=1e-8
    )
    scheduler = th.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)
    gamma = 0.99  # Sconto per reward futura

    cumulative_rewards = []  # Reward cumulativa per ogni episodio
    for episode in range(n_episodes):
        obs = env.reset()
        prev_inventory = {key: 0 for key in MATERIAL_REWARDS.keys()}
        
        jump_window = deque(maxlen=20)
        inventory_reward_given = False
        isInventoryOpen = False
        
        cumulative_episode_reward = 0
        episode_rewards = []  # Salva le reward per ogni passo
        steps = []  # Salva i passi per il grafico

        episode_dir = f"./data/Stats/RLTranding_reward/episode{episode + 1}"
        os.makedirs(episode_dir, exist_ok=True)

        batch_rewards = []
        batch_log_probs = []
        batch_advantages = []

        for step in range(max_steps):
            action = agent.get_action(obs)
            action["ESC"] = 0
            obs, _, done, _ = env.step(action)

            # Aggiungi un'esplorazione casuale
            if np.random.rand() < 0.10: # 10% di probabilità di azione casuale
                action = env.action_space.sample()

            # Aggiorna la finestra dei salti
            jump_window.append(action.get("jump", 0))
    
            # Calcola la reward basata sull'inventario
            inventory = obs["inventory"]
            material_reward = compute_reward(inventory, prev_inventory)
            action_reward, inventory_reward_given = action_based_reward(action, inventory, jump_window, inventory_reward_given)
            reward = material_reward + action_reward
            # Accumula la reward per valutare l'episodio
            cumulative_episode_reward += reward

             # Aggiorna lo stato dell'inventario
            if action.get("inventory", 0) == 1:
                isInventoryOpen = not isInventoryOpen
                if not isInventoryOpen:
                    inventory_reward_given = False  # Reset quando l'inventario si chiude

            # Aggiorna i dati per il grafico
            steps.append(step + 1)
            episode_rewards.append(cumulative_episode_reward)

            # Aggiorna il grafico interattivo
            print(f"step: {step} reward: {reward}")
            line.set_xdata(steps)
            line.set_ydata(episode_rewards)
            ax.relim()
            ax.autoscale_view()
            fig.canvas.draw_idle()
            plt.pause(0.01)

            prev_inventory = {key: inventory.get(key, 0) for key in MATERIAL_REWARDS.keys()}

            # Ottieni la distribuzione e l'azione trasformata
            da = agent.policy.get_output_for_observation(
                agent._env_obs_to_agent(obs),
                agent.policy.initial_state(1),
                th.tensor([False])
            )[0]
            ac = agent._env_action_to_agent(action, to_torch=True, check_if_null=False)

            # Calcola il log_prob e la perdita
            log_prob = agent.policy.get_logprob_of_action(da, ac)
            advantage = reward + gamma * log_prob.detach().mean() - log_prob.mean()
            advantage = normalize(advantage)

            # Accumula i valori batch-wise
            batch_rewards.append(reward)
            batch_log_probs.append(log_prob)
            batch_advantages.append(advantage)

            epsilon = 0.2  # Parametro di clipping (valore tipico: 0.1-0.3)

            if reward != 0 and (step + 1) % 10 == 0:
                cumulative_reward = sum(batch_rewards)
                # Converte in tensori
                batch_log_probs = th.stack(batch_log_probs)
                batch_advantages = th.stack(batch_advantages)

                print(f"Reward cumulativa ultimi 10 passi: {cumulative_reward} \n")

                # Calcolo del rapporto r_t e applicazione del clipping
                r_t = th.exp(batch_log_probs - batch_log_probs.detach())
                clipped_ratio = th.clamp(r_t, 1 - epsilon, 1 + epsilon)
                loss = -th.min(r_t * batch_advantages, clipped_ratio * batch_advantages).mean()

                # Backpropagation e aggiornamento
                optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(agent.policy.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step() # Aggiorna il learning rate dopo ogni batch

                # Reset dei batch
                batch_rewards = []
                batch_log_probs = []
                batch_advantages = []

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

    # Grafico complessivo
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, n_episodes + 1), cumulative_rewards, marker='o', label="Reward Cumulativa Totale")
    plt.title("Andamento della Reward Cumulativa per Tutti gli Episodi", fontsize=14)
    plt.xlabel("Episodio", fontsize=12)
    plt.ylabel("Reward Cumulativa", fontsize=12)
    plt.grid(True)
    plt.legend()
    plt.savefig("./data/Stats/RLTranding_reward/cumulative_reward_trend_all_episodes.png")
    plt.show()

    state_dict = agent.policy.state_dict()
    th.save(state_dict, "ppo_updated_weights_rmsprop.weights")

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
