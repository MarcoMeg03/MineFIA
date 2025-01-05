from gym.envs.registration import register
from minerl.env import _singleagent
from minerl.herobraine.env_specs.basalt_specs import BasaltTimeoutWrapper, DoneOnESCWrapper
from env.FIAenv import FIATreechopEnvSpec

# Funzione di entry point per creare l'ambiente
def fia_treechop_entrypoint():
    env_spec = FIATreechopEnvSpec()
    env = _singleagent._SingleAgentEnv(env_spec=env_spec)
    env = BasaltTimeoutWrapper(env)
    env = DoneOnESCWrapper(env)
    return env

# Registra l'ambiente FIA-Treechop-v0
register(
    id='FIA-Treechop-v0',
    entry_point='register_envs:fia_treechop_entrypoint',
)
