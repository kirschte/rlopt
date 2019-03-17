#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module file sets up the base environment in accordance to OpenAI's `gym`:
**action enumeration**.

@author: Moritz Kirschte
@date Mar 2019
"""

from enum import Enum, unique


@unique
class Action(Enum):
    """Enumeration of all possible actions:

    @id:0 TERMINATE. Intend to terminate the application.
    @id:1 INCI. Intend to increases the first bound variable about 1.
    @id:2 INCJ. Intend to increases the second bound variable about 1.
    @id:3 RESETI. Intend to reset the first bound variable to zero.
    @id:4 RESETJ. Intend to reset the second bound variable to zero.
    @id:5 SWAP. Intend to swap the list elements i and j.
    """
    TERMINATE = 0
    INCI = 1
    INCJ = 2
    RESETI = 3
    RESETJ = 4
    SWAP = 5
