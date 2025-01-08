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
from minerl.herobraine.hero.mc import ALL_ITEMS
from minerl.herobraine.hero.handler import Handler


from minerl.herobraine.env_specs.human_controls import HumanControlEnvSpec

MINUTE = 20 * 60  # 20 tick per secondo * 60 secondi


class FIATreechopEnvSpec(BasaltBaseEnvSpec):
    """

    """

    def __init__(self):
        super().__init__(
            name="FIA-Treechop-v0",
            demo_server_experiment_name="treechop",
            max_episode_steps=3*MINUTE,
            preferred_spawn_biome="forest",
            inventory=[
                #dict(type="oak_log", quantity=5),
            ],
        )
    def create_mission_handlers(self):
        # Aggiunge gestori per osservazioni e azioni pertinenti
        return super().create_mission_handlers() + [
            handlers.InventoryObservation(["log", "planks", "crafting_table"]),
            handlers.CraftingAction(["planks", "crafting_table"]),
            handlers.FlatInventoryObservation(["log", "planks", "crafting_table"]),
        ]

    def determine_success_from_rewards(self, rewards: List[float]) -> bool:
        # Definisce il successo se il banco da lavoro è costruito
        return "crafting_table" in self.current_observation.get("inventory", {}) and \
               self.current_observation["inventory"]["crafting_table"] >= 1

    def create_observables(self) -> List[Handler]:
        return [
            handlers.POVObservation(self.resolution),
            handlers.FlatInventoryObservation(ALL_ITEMS)
        ]


    @staticmethod
    def fia_treechop_entrypoint():
        from env.FIAenv import FIATreechopEnvSpec  # Assicurati che il percorso sia corretto
        env_spec = FIATreechopEnvSpec()
        env = _singleagent._SingleAgentEnv(env_spec=env_spec)
        env = BasaltTimeoutWrapper(env)
        env = DoneOnESCWrapper(env)
        return env