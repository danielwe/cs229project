#!/usr/bin/env python

"""
Implement keras callback classes that log the state at each time step.

Adapted from `keras-rl`s `TrainEpisodeLogger`.

"""

from keras.callbacks import Callback


class RewardsLogger(Callback):
    def __init__(self):
        # Some algorithms compute multiple episodes at once since they are
        # multi-threaded. We therefore use a dictionary that is indexed by the
        # episode to separate episodes from each other.
        self.rewards = {}
        self.step = 0

    def on_episode_begin(self, episode, logs):
        self.rewards[episode] = []

    def on_step_end(self, step, logs):
        episode = logs['episode']
        self.rewards[episode].append(logs['reward'])
        self.step += 1

    def __getstate__(self):
        return {key: getattr(self, key)
                for key in ('rewards', 'step')}


class ObservationsLogger(RewardsLogger):
    def __init__(self):
        RewardsLogger.__init__(self)
        self.observations = {}
        self.actions = {}

    def on_episode_begin(self, episode, logs):
        RewardsLogger.on_episode_begin(self, episode, logs)
        self.observations[episode] = []
        self.actions[episode] = []

    def on_step_end(self, step, logs):
        RewardsLogger.on_step_end(self, step, logs)
        episode = logs['episode']
        self.observations[episode].append(logs['observation'])
        self.actions[episode].append(logs['action'])

    def __getstate__(self):
        state = RewardsLogger.__getstate__(self)
        state.update({key: getattr(self, key)
                      for key in ('observations', 'actions')})
        return state
