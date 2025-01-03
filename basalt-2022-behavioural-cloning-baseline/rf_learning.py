from argparse import ArgumentParser
import pickle

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

def main(model, weights, env, n_episodes=3, max_steps=int(1e9), show=True):
    env = aicrowd_gym.make(env)
    agent_parameters = pickle.load(open(model, "rb"))
    policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
    agent = MineRLAgent(env, policy_kwargs=policy_kwargs, pi_head_kwargs=pi_head_kwargs)
    agent.load_weights(weights)

    # Congela tutti i parametri del modello
    for param in agent.policy.parameters():
        param.requires_grad = False
        
    # Sblocca i parametri del `pi_head` (responsabili della distribuzione delle azioni)
    for param in agent.policy.pi_head.parameters():
        param.requires_grad = True

    # Istanziamento dell'ottimizzatore
    optimizer = th.optim.RMSprop(
        filter(lambda p: p.requires_grad, agent.policy.parameters()), 
        lr=0.00001, 
        alpha=0.99, 
        eps=1e-8
    )
    
    # Istanziamento dello scheduler
    scheduler = th.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)
 
    gamma = 0.99  # Sconto per reward futura

    for _ in range(n_episodes):
        obs = env.reset()
        prev_inventory = {key: 0 for key in MATERIAL_REWARDS.keys()}
        
        jump_window = deque(maxlen=20)  # Finestra mobile per i salti
        inventory_reward_given = False  # Flag per la reward di inventario
        isInventoryOpen = False
        
        cumulative_episode_reward = 0
        cumulative_reward = 0
        
        batch_rewards = []
        batch_log_probs = []
        batch_advantages = []

        for step in range(max_steps):
            action = agent.get_action(obs)
            action["ESC"] = 0
            obs, _, done, _ = env.step(action)
            
            # Aggiungi un'esplorazione casuale
            if np.random.rand() < 0.10:  # 10% di probabilità di azione casuale
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
                
            prev_inventory = {key: inventory.get(key, 0) for key in MATERIAL_REWARDS.keys()}

            # Ottieni la distribuzione e l'azione trasformata
            output = agent.policy.get_output_for_observation(
                agent._env_obs_to_agent(obs),
                agent.policy.initial_state(1),
                th.tensor([False])
            )

            pd = output[0]  # Distribuzione delle probabilità
            ac = agent._env_action_to_agent(action, to_torch=True, check_if_null=False)

            # Calcola il log_prob e la perdita
            log_prob = agent.policy.get_logprob_of_action(pd, ac)
            advantage = reward + gamma * log_prob.detach().mean() - log_prob.mean()
            advantage = normalize(advantage)
            
            # Accumula i valori batch-wise
            batch_rewards.append(reward)
            batch_log_probs.append(log_prob)
            batch_advantages.append(advantage)

            epsilon = 0.2  # Parametro di clipping (valore tipico: 0.1-0.3)

            if reward != 0 and (step + 1) % 10 == 0:
                # Reward cumulativa
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
                scheduler.step()  # Aggiorna il learning rate dopo ogni batch

                # Reset dei batch
                batch_rewards = []
                batch_log_probs = []
                batch_advantages = []
                
            elif (step + 1) % 10 == 0:
                # Reset dei batch
                batch_rewards = []
                batch_log_probs = []
                batch_advantages = []
                
            if show:
                env.render()
            if done:
                break
                
        print(f"Reward cumulativa per l'episodio: {cumulative_episode_reward}")
        
    env.close()

    # Salva i pesi aggiornati
    with open("ppo_updated_weights_rmsprop.weights", "wb") as f:
        pickle.dump(agent.policy.state_dict(), f)


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
