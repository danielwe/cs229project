#!/usr/bin/env python

"""Plot training and test data."""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


def plot_observations(logger, n_episodes=2):
    shoulder_targets = []
    shoulder_states = []
    elbow_targets = []
    elbow_states = []
    for ep in range(n_episodes):
        for obs in logger.observations[ep]:
            shoulder_targets.append(obs[0])
            shoulder_states.append(obs[2])
            elbow_targets.append(obs[1])
            elbow_states.append(obs[3])

    _, ax = plt.subplots(nrows=2, sharex=True)
    ax[0].plot(shoulder_targets)
    ax[0].plot(shoulder_states)
    ax[0].set_ylabel('shoulder$ \ / \ \mathrm{rad}$')
    ax[1].plot(elbow_targets)
    ax[1].plot(elbow_states)
    ax[1].set_ylabel('elbow$ \ / \ \mathrm{rad}$')
    ax[1].set_xlabel('$t \ / \ \mathrm{steps}$')
    return ax


def plot_reward_history(logger, axes=None, label='', size=20):
    rew = rewards(logger)
    cs = np.cumsum(rew)
    mean = (cs[size:] - cs[:-size]) / size

    if axes is None:
        _, axes = plt.subplots()
    t = list(range(len(rew)))
    #axes.plot(t, rew, label=label)
    axes.plot(t[size // 2:-size // 2], mean, label=label)
    axes.set_ylabel('reward')
    axes.set_xlabel('$t \ / \ \mathrm{episodes}$')
    return axes


def plot_reward_statistics(loggers, axes=None):
    data = pd.DataFrame({
        key: rewards(value, bs=100) for key, value in loggers.items()
    })
    return sns.boxplot(data=data, ax=axes)


def rewards(logger, bs=None):
    rew = []
    if bs is None:
        bs = len(next(iter(logger.rewards.values())))
    for _, r in sorted(logger.rewards.items()):
        for i in range(len(r) // bs):
            rew.append(np.mean(r[bs * i: bs * (i + 1)]))
    return rew
