#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module sets up the base environment in accordance to OpenAI's `gym` including
- action enumeration,
- sorting environment acting on and
- histogram wrapper for better exploration.

@author: Moritz Kirschte
@date Mar 2019
"""

from .action import Action
from .env import RLSort
from .histogram import HistWrapper
