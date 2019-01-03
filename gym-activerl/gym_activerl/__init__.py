import logging
from gym.envs.registration import register
logger = logging.getLogger(__name__)
register(
    id='activerl-v0',
    entry_point='gym_activerl.envs:ActiverlEnv',
)

