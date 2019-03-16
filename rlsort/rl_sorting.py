#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This program represents the unique and hilarious RLSort algorithm.
Inspired by White, Martinez and Rudolph in
`Generating a Novel Sort Algorithm using Reinforcement Programming` from 2010
this algorithm essentially learns how to sort a list with tabular Q-learning
(although any other variable optimization function could also be learned)
and does after some tuning generalize to all different kind of lists well.

NB that the state and action space is completely independent of the concrete
list itself and its length.

Moreover, a reconstruction of the tabular towards a real program-logic is
successfully investigated where the q-table is shortened to exactly those
transition functions which are needed to complete sorting.

Finally a performance comparison between well-known sorting algorithms
like Bubblesort or Selectionsort is laid out.

@author: Moritz Kirschte
@date Mar 2019
"""

import argparse

import numpy as np
import pandas as pd

from env import Action, RLSort, HistWrapper


def train(list_len,
          Q=None,
          epochs=5_000_000,
          n_iter=2000,
          greedy_steps=50_000,
          stop_hist_exploration=0.9,
          q_decay=0.95,
          lr=.1,
          y=.7,
          render=10_000):
    """
    Trains on the RL-environment with tabular Q-learning.

    # Arguments:
        list_len: int[3..]. Determines the maximal list length
                            of the to-be-trained permutations.
        Q: np.array. A pre-defined Q-table.
                     If None a new one will be generated.
        epochs: int[1..]. To be trained epochs.
        n_iter: int[1..]. Length of one episode.
        greedy_steps: int[1..]. Initial rate at which
                                the greedy factor will be halved
                                and this factor doubled.
        stop_hist_exploration: float[0..1]. Percentage at which episode
                                            histogram exploration should be stopped.
        q_decay: float[0..1]. Decay factor of the Q-table. Only applied
                              after `stop_hist_exploration` limit is reached
                              and one permutation cycle goes through.
        lr: float[0..1]. Learning rate.
        y: float[0..1]. Discount factor of Bellman equation.
        render: int[0..]. Cycle length of when the resulting state
                          of one episode should be rendered.
                          If zero no rendering is applied.

    # Returns
        final trained Q-table of shape (n_states, n_actions).
    """
    greedy = 1
    env = HistWrapper(RLSort(list_len))
    if Q is None:
        Q = np.zeros((env.observation_space.high[0], env.action_space.n))
    print('Q-Table Values of shape %s:' % str(Q.shape))

    for eps in range(0, epochs):
        # decrease greedy-exploration through time
        if eps > 0 and eps % greedy_steps == 0:
            greedy /= 2
            greedy_steps *= 2

        s = env.reset()
        its = 0
        rews = 0
        while True:
            # explore greedily
            if np.random.rand() < greedy:
                a = np.random.randint(env.action_space.n)
            else:
                a = np.argmax(Q[s, :])
            # learn Q-table
            next_s, rew, done, _ = (env.step(a)
                                    if eps < epochs * stop_hist_exploration
                                    else env.env.step(a))
            Q[s, a] = (1 - lr)*Q[s, a] + lr*(rew + y*np.max(Q[next_s, :]))
            s = next_s
            # Exit criteria
            rews += rew
            its += 1
            if done or its >= n_iter:
                break

        if render > 0 and eps % render == 0:
            print('> {:6,} Episode: {:5.0f} reward'.format(eps, rews))
            env.render(mode='ascii')
        if eps >= stop_hist_exploration * epochs and eps % len(env.env._perm) == 0:
            Q *= q_decay

    return Q


def test(Q, list_len, n_iter=2000):
    """Tests the trained Q-table with every possible permutation (<list_len)
    iteratively by printing its resulting state / reward.

    Furthermore, it stores every reachable state by a whole transition frame.

    # Arguments
        Q: np.array. A pre-trained Q-table.
        list_len: int[3..]. Determines the maximal list length
                            of the to-be-tested permutations.
        n_iter: int[1..]. Length of one episode.

    # Returns
        Dataframe containing each unique reached state-action-next_state transition.
        Could be grouped by its states afterward in order to investigate
        the best-taken action in that state and the following reached next states.
    """
    env = RLSort(list_len)
    buffer = pd.DataFrame(columns=['state',
                                   'next state',
                                   'to-be-executed action',
                                   'last action',
                                   'li_lj',
                                   'i_j',
                                   'i',
                                   'j'])
    for perm in env._perm:
        s = env._reset(np.array(perm))
        its = 0
        rews = 0
        buffer_ = []  # for performance reasons: list as working object
        while True:
            a = np.argmax(Q[s, :])

            li_lj, i_j, i, j, _ = env._state()
            buffer_.append({'state': s,
                            'next state': -1,
                            'to-be-executed action': Action(a).name,
                            'last action': env._last_action.name,
                            'li_lj': li_lj,
                            'i_j': i_j,
                            'i': i,
                            'j': j})
            s, rew, done, _ = env.step(a)
            buffer_[-1]['next state'] = s

            # Exit criteria
            rews += rew
            its += 1
            if done or its >= n_iter:
                print('> {:} Episode: {:4.0f} reward ({:})'.format(perm,
                                                                   rews,
                                                                   env._list))
                break
        buffer = buffer.append(buffer_, ignore_index=True)
    return buffer.drop_duplicates().reset_index(drop=True)


def print_qtable(Q):
    """Prints a Q-table nicely formatted.

    # Arguments
        Q: np.array(n_states, n_actions). To-be-printed Q-table.
    """
    print('Final Q-table values of shape %s:' % str(Q.shape))
    print('(ID)\t%7s %7s %7s %7s %7s %7s' % ('TERM',
                                             'I++',
                                             'J++',
                                             'RESETI',
                                             'RESETJ',
                                             'SWAP'))
    for idx, arr in enumerate(Q[:-1]):
        print('(%3d)\t%7.2f %7.2f %7.2f %7.2f %7.2f %7.2f' % (idx, *arr))


def reconstruct_algorithm(WSK=None, filename='transitions.csv'):
    """Recontruct efficient algorithm out of reduced transitions.

    # Arguments
        WSK: pd.DataFrame. Wertschöpfungskette AKA transitions.
                           If None loaded from disk, ow saved to disk.
        filename: str. The to-be-saved or to-be-loaded location on disk.

    # Returns
        Callable function with the to-be-sorted list as a parameter.
    """
    if WSK is None:
        WSK = pd.read_csv(filename)
    else:
        WSK.to_csv(filename)

    unique_transitions = WSK.drop(['state', 'next state'], axis=1) \
                            .drop_duplicates() \
                            .reset_index(drop=True)
    unique_transitions = unique_transitions.set_index(['li_lj',
                                                       'i_j',
                                                       'i',
                                                       'j',
                                                       'last action'])
    unique_transitions = unique_transitions['to-be-executed action']
    print('{} unique transitions:'.format(len(unique_transitions)))
    print(WSK.groupby(['li_lj', 'i_j', 'i', 'j', 'last action']).agg(
        {'to-be-executed action': 'max',
         'state': 'max',
         'next state': lambda x: tuple(x)}
    ))

    def _algorithm(lst, transitions, lst_len, i=0, j=0, l_ac=Action.TERMINATE):
        """The final reconstructed sorting algorithm.
        Arguments:
            lst: np.array. To be sorted list.
            transitions: pd.Series. Unique transitions used as a lookup table.
            lst_len: int. Length of `list`. Used for algorithm speedup.
            i: int. Inbound variable for algorithm functionality.
            j: int. Another inbound variable for algorithm functionality.
            l_ac: Action. The last performed action for algorithm functionality/execution.
        Returns:
            The sorted list and the number of steps needed for sorting.
        """
        steps = 0
        while True:
            state = (
                0 if i >= lst_len or j >= lst_len or lst[i] <= lst[j] else 1,
                RLSort.compare(i, j),
                -1 if i == 0 else 1 if i == lst_len else 0,
                -1 if j == 0 else 1 if j == lst_len else 0,
                l_ac.name
            )

            l_ac = Action[transitions.loc[state]]
            steps += 1
            if l_ac is Action.INCI:
                i += 1
            if l_ac is Action.INCJ:
                j += 1
            if l_ac is Action.RESETI:
                i = 0
            if l_ac is Action.RESETJ:
                j = 0
            if l_ac is Action.SWAP:
                lst[[i, j]] = lst[[j, i]]
            if l_ac is Action.TERMINATE:
                break

        return lst, steps

    return lambda l: _algorithm(np.array(l), unique_transitions, len(l))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Sorting through Reinforcement Learning.'
    )
    parser.add_argument('list', metavar='N', type=int, nargs='*',
                        help='to-be-sorted list elements',
                        default=[6, 4, 3, 7, 0])
    parser.add_argument('--train', action='store_true',
                        help='whether to train or not')
    parser.add_argument('--filename', '-f', metavar='FILE',
                        default='transitions.csv',
                        help='location of the learned algorithm')
    args = parser.parse_args()

    WSK = None  # If not specified -> read from disk
    if args.train:
        LIST_LEN = 6
        Q = train(LIST_LEN)
        WSK = test(Q, LIST_LEN + 1)
        print_qtable(Q)

    algorithm = reconstruct_algorithm(WSK, filename=args.filename)

    # Example call:
    print('Sorted list in {1:} steps: {0:}'.format(*algorithm(args.list)))
