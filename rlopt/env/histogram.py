#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module file sets up the base environment in accordance to OpenAI's `gym`:
**histogram wrapper for better exploration**.

@author: Moritz Kirschte
@date Mar 2019
"""

import numpy as np
from gym import RewardWrapper


class HistWrapper(RewardWrapper):
    """Wrapps an environment with a histogram exploration
    (currently, just state-spaced supported).
    Any state gets a bonus proportional to `1/sqrt(n)`
    where `n` denotes how often that particular state was visited before.
    This makes new states more likely to be visited.

    # Arguments
        env: gym.Env. The to be wrapped env.
        bonus_coeff: float[0..]. Multiplier of the bonus term.
                            Determines how intensive should be explored.
    """
    def __init__(self, env, bonus_coeff=50):
        super(HistWrapper, self).__init__(env)
        assert bonus_coeff >= 0, "Only a positive bonus coefficient is reasonable."
        self._hist = np.zeros(self.observation_space.high[0])
        self._bonus_coeff = bonus_coeff

    def reward(self, reward):
        count = self._hist[self.env.state]
        reward += self._bonus_coeff / np.sqrt(count + 1e-8)
        self._hist[self.env.state] += 1
        return reward
