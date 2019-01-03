import logging
from gym.envs.registration import register
logger = logging.getLogger(__name__)
register(
    id='activerl1-v0',
    entry_point='gym_activerl1.envs:ActiverlEnv1',
)

