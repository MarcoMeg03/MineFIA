from typing import List, Optional, Sequence

import gym

from minerl.env import _fake, _singleagent
from minerl.herobraine import wrappers
from minerl.herobraine.env_spec import EnvSpec
from minerl.herobraine.env_specs import simple_embodiment
from minerl.herobraine.hero import handlers, mc
from minerl.herobraine.env_specs.basalt_specs import BasaltBaseEnvSpec
from minerl.env import _singleagent
from minerl.herobraine.env_specs.basalt_specs import BasaltTimeoutWrapper, DoneOnESCWrapper

from minerl.herobraine.env_specs.human_controls import HumanControlEnvSpec

MINUTE = 20 * 60  # 20 tick per secondo * 60 secondi


class FIATreechopEnvSpec(BasaltBaseEnvSpec):
    """
    .. image:: ../assets/fia/treechop.gif
      :scale: 100 %
      :alt:
    
    Dopo essere spawnato in una foresta, l'obiettivo è abbattere almeno 5 alberi. 
    Ogni albero deve essere completamente rimosso (tutti i blocchi di tronco tagliati).
    Termina l'episodio impostando l'azione "ESC" a 1.
    """

    def __init__(self):
        super().__init__(
            name="FIA-Treechop-v0",
            demo_server_experiment_name="treechop",
            max_episode_steps=3*MINUTE,
            preferred_spawn_biome="forest",
            inventory=[
                dict(type="iron_axe", quantity=1),
                dict(type="stone_pickaxe", quantity=1),
                dict(type="oak_sapling", quantity=10),
                dict(type="bone_meal", quantity=32),
            ],
        )

    def fia_treechop_entrypoint():
        from env.FIAenv import FIATreechopEnvSpec  # Assicurati che il percorso sia corretto
        env_spec = FIATreechopEnvSpec()
        env = _singleagent._SingleAgentEnv(env_spec=env_spec)
        env = BasaltTimeoutWrapper(env)
        env = DoneOnESCWrapper(env)
        return env