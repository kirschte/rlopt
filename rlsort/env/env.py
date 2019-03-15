#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module file sets up the base environment in accordance to OpenAI's `gym`:
**sorting environment acting on**.

@author: Moritz Kirschte
@date Mar 2019
"""

from itertools import permutations
import random

import numpy as np
from gym import Env
from gym import spaces
from gym.utils import seeding

from . import Action

class RLSort(Env):
    """The environment representation of the RL algorithms.
    Capsuled into an OpenAI Gym environment due to a standardized
    method definitions and further compatibility with other algorithms.

    # Arguments
        list_len: int[3..]. Specifies the maximum list length
                            to generate permutations from.
    """
    def __init__(self, list_len=6):
        super(RLSort, self).__init__()
        assert list_len > 2, "Too small list length for training"

        # Data-specific variables
        self._list = None
        self._len = None

        # RP-specific variables
        self._i = None
        self._j = None
        self._last_action = Action.TERMINATE

        # Setting up all possible permutations for fair reseting
        # Length: sum_j[3..len]( prod_i[1..j](i) )
        self._perm = {}
        for l_len in range(3, list_len + 1):
            self._perm.update(
                {v: k for k, v in enumerate(permutations(np.arange(0, l_len)))}
            )

        # Inherited necessary definitions
        self.observation_space = spaces.Box(0,
                                            2 * 3**3 * len(Action),
                                            shape=(1,),
                                            dtype=np.int32)
        self.action_space = spaces.Discrete(len(Action))
        self.reward_range = (-300., 300.)
        self.seed()

    @staticmethod
    def compare(a, b):
        """Compares like JAVA's `compareTo` method.

        # Arguments
            a: int. Parameter A.
            b: int. Parameter B.
        # Returns
            -1: If A < B
             0: If A == B
             1: if A > B
        """
        return int(a > b) - int(a < b)

    @staticmethod
    def is_sorted(arr):
        """Checks if list is sorted (used for reward determination).

        # Arguments
            arr: List[int]. The to be checked list.
        # Returns
            True: If `arr` is sorted.
            False: otherwise.
        """
        return all(a <= b for a, b in zip(arr, arr[1:]))

    @property
    def _out_bounded(self):
        """Checks of the bound-variables are truly inbound of the list.

        # Returns
            True: If `i` and `j` are inbound
            False: otherwise.
        """
        return self._i >= self._len or self._j >= self._len

    @property
    def state(self):
        """Maps the RP-specific state (see `_state`)
        to a unique numerical identifier.

        # Returns
            The unique RP-state number.
            Anything between 0 and `observation_space.high`
        """
        li_lj, i_j, i, j, l_ac = self._state()
        return (li_lj) + \
            2 * (i_j + 1) + \
            2 * 3 * (i + 1) + \
            2 * 3**2 * (j + 1) + \
            2 * 3**3 * (l_ac)

    def step(self, action):
        rew = 0
        term = False
        self._last_action = Action(action)

        if Action(action) is Action.TERMINATE:
            term = True
            rew += 300 if self.is_sorted(self._list) else -300
        elif Action(action) is Action.INCI:
            self._i = min(self._i + 1, self._len)  # for safety
        elif Action(action) is Action.INCJ:
            self._j = min(self._j + 1, self._len)  # for safety
        elif Action(action) is Action.RESETI:
            self._i = 0
        elif Action(action) is Action.RESETJ:
            self._j = 0
        elif Action(action) is Action.SWAP:
            if self._out_bounded:
                rew -= 10
            else:  # everything's fine and swapable
                self._list[[self._i, self._j]] = self._list[[self._j, self._i]]

                call = np.int64.__lt__ if self._i < self._j else np.int64.__gt__
                rew += 10 if call(self._list[self._i], self._list[self._j]) else -10

        return self.state, rew, term, None

    def reset(self):
        selection = np.array(random.choice(list(self._perm.keys())))
        return self._reset(selection)

    def render(self, mode='ascii'):
        if mode == 'ascii':
            print(self._list)
        else:
            super(RLSort, self).render(mode=mode)  # just raise an exception

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        random.seed(seed)
        return [seed]

    def _reset(self, arr):
        """Resets the RP- & Data-specific environment given a list.

        # Arguments
            arr: List[int]. The new list in our environment.
        # Returns
            state: int. The current new state.
        """
        self._list = arr
        self._len = len(arr)
        self._i = 0
        self._j = 0
        self._last_action = Action.TERMINATE
        return self.state

    def _state(self):
        """Calculates for each RP-specfic characteristica its corresponding value.

        # Returns
            Tupel[int, int, int, int, int].
            id:0 `l[i] vs. l[j]`. le: 0. gt: 1.
            id:1 `i vs. j`. lt: -1. eq: 0. gt: 1.
            id:2 `i`. ==0: -1. ==list_len: 1. ow: 0.
            id:3 `j`. ==0: -1. ==list_len: 1. ow: 0.
            id:4 `l_ac`. `Action`-Enum value of specific last action `l_ac`.
        """
        return 0 if self._out_bounded or self._list[self._i] <= self._list[self._j] else 1, \
            self.compare(self._i, self._j), \
            -1 if self._i == 0 else 1 if self._i == self._len else 0, \
            -1 if self._j == 0 else 1 if self._j == self._len else 0, \
            self._last_action.value
