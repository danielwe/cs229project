#!/usr/bin/env python

"""Script to train the original model in the modified environment."""

from environments import ModifiedArmEnv

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, concatenate
from keras.optimizers import Adam

from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess

from loggers import RewardsLogger, ObservationsLogger

import sys
import pickle


def pickledump(obj, fname):
    with open(fname, 'wb') as f:
        return pickle.dump(obj, f)


def pickleload(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)


def file_prefix(sigma, learning_rate):
    return "model1_sigma-{}_lr-{}".format(sigma, learning_rate)


def main(args):
    sigma, learning_rate, file_prefix = args

    env = ModifiedArmEnv(visualize=False)
    input_shape = (1, ) + env.observation_space.shape
    nb_actions = env.action_space.shape[0]

    # Create actor and critic networks
    actor = Sequential()
    actor.add(Flatten(input_shape=input_shape))
    actor.add(Dense(32))
    actor.add(Activation('relu'))
    actor.add(Dense(32))
    actor.add(Activation('relu'))
    actor.add(Dense(32))
    actor.add(Activation('relu'))
    actor.add(Dense(nb_actions))
    actor.add(Activation('sigmoid'))

    action_input = Input(shape=(nb_actions,), name='action_input')
    observation_input = Input(shape=input_shape, name='observation_input')
    flattened_observation = Flatten()(observation_input)
    x = concatenate([action_input, flattened_observation])
    x = Dense(64)(x)
    x = Activation('relu')(x)
    x = Dense(64)(x)
    x = Activation('relu')(x)
    x = Dense(64)(x)
    x = Activation('relu')(x)
    x = Dense(1)(x)
    x = Activation('linear')(x)
    critic = Model(inputs=[action_input, observation_input], outputs=x)

    # Set up the agent for training
    memory = SequentialMemory(limit=100000, window_length=1)
    random_process = OrnsteinUhlenbeckProcess(
        theta=.15, mu=0., sigma=sigma, dt=env.stepsize, size=env.noutput)
    agent = DDPGAgent(
        nb_actions=nb_actions,
        actor=actor,
        critic=critic,
        critic_action_input=action_input,
        memory=memory,
        nb_steps_warmup_critic=100,
        nb_steps_warmup_actor=100,
        random_process=random_process,
        gamma=.99,
        target_model_update=1e-3,
        delta_clip=1.,
    )
    agent.compile(Adam(lr=learning_rate, clipnorm=1.), metrics=['mae'])

    # Train the model
    training_history = RewardsLogger()
    env.reset()
    agent.fit(
        env,
        nb_steps=100000,
        visualize=False,
        verbose=1,
        nb_max_episode_steps=200,
        log_interval=10000,
        callbacks=[training_history],
    )

    # Save weights and training history
    agent.save_weights(file_prefix + '_weights.h5f', overwrite=True)
    pickledump(training_history, file_prefix + '_training_history.pkl')

    # Set test parameters
    test_nb_episodes = 10
    test_nb_max_episode_steps = 1000

    # Run test
    test_history = ObservationsLogger()
    env.reset()
    agent.test(
        env,
        nb_episodes=test_nb_episodes,
        visualize=False,
        nb_max_episode_steps=test_nb_max_episode_steps,
        callbacks=[test_history],
    )
    # Save test history
    pickledump(test_history, file_prefix + '_test_history.pkl')


if __name__ == '__main__':
    sigma, learning_rate = sys.argv[1:3]
    main((sigma, learning_rate, file_prefix(sigma, learning_rate)))
