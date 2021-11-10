import ast
import json
import logging
import os
import random
import numpy as np
from abc import abstractmethod
from collections import deque
from importlib.util import spec_from_file_location, module_from_spec
from rgkit import rg
from rgkit import game as rg_game
from rgkit.settings import settings
from tensorflow.keras.models import load_model

np.set_printoptions(precision=3, suppress=True)

# noinspection SpellCheckingInspection
with open('rgkit/maps/default.py') as _:
    settings.init_map(ast.literal_eval(_.read()))

# TODO: compute action preferences once per turn not once per robot per turn.
# TODO: reuse state_array and others for inference not just training
# TODO: compare performance of model(x) vs model.predict(x).

# TODO: load an existing model directory


class DRLRobot:
    MAX_ROBOTS = 50

    def __init__(self, model_dir='.', memory_size=2000, exploit=True, epsilon_decay=0.99, mini_batch_size=1000,
                 **model_params):
        self.model_dir = model_dir
        self.model_file = os.path.join(model_dir, 'model.h5')

        # load or build model
        if os.path.isfile(self.model_file):
            print(f'Loading {self.model_file}')
            self.load()
        else:
            self.model = self._build_model(**model_params)

        # create history of action results
        self.memory = deque(maxlen=memory_size)

        assert mini_batch_size > self.MAX_ROBOTS, 'mini batch size must be greater than maximum number of robots.'

        # preallocate arrays for efficient computation
        input_size = self.model.input_shape[1:]
        action_size = self.model.output_shape[1]
        self.mini_batch_size = mini_batch_size
        state_size = tuple([mini_batch_size] + list(input_size))
        self.state_array = np.zeros(state_size, dtype=np.float32)
        self.action_array = np.empty(mini_batch_size, dtype=int)
        self.target_array = np.empty(mini_batch_size, dtype=np.float32)
        self.next_state_array = np.zeros(state_size, dtype=np.float32)
        self.done_array = np.empty(mini_batch_size, dtype=bool)
        self.priority_actions_array = np.empty((self.MAX_ROBOTS, action_size), dtype=int)

        # set exploration/exploitation parameters
        self.exploit = exploit
        self.epsilon = 1.0
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = 0.01

        # start exponential running average
        self.ema_reward = 0  # exponential moving average reward
        self.ema_alpha = 0.0001  # ema coefficient

        # set discount factor for q-learning
        self.gamma = 0.95

        # initialize fields set by original Game.
        self.location = None
        self.robot_id = None
        self.player_id = None
        self.hp = None

        # initialize new fields set and retained by Game.
        self.state = None
        self.action = None

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def percept(self, game, robot):
        """
        Return the current state, reward, and whether or not the robot is done.
        :param game:
        :param robot:
        :return:
        """
        reward = self.get_reward(game, robot)
        done = robot.hp <= 0 or game.turn == 99
        if not done:
            state = self.get_state(game, robot)
            if len(state.shape) < 2 or state.shape[0] != 1:
                state = state[None, :]
        else:
            state = None
        return state, reward, done

    @staticmethod
    def is_valid(action, robot):
        return action[0] in ['guard', 'suicide'] or action[1] in rg.locs_around(
            robot.location, filter_out=['invalid', 'obstacle']
        )

    def get_valid_action(self, priority_action_indexes, game, robot):
        for next_action_index in priority_action_indexes:
            next_action = self.get_action(next_action_index, game, robot)
            if self.is_valid(next_action, robot):
                break
        else:
            raise Exception('No valid actions.')
        return next_action_index, next_action

    def act(self, game):
        # check the zombies first because another robot might be in this robot's original position.
        robot = game.zombies.get(self.robot_id, game.robots.get(self.location))
        if robot.next_action is None:
            action_size = self.model.output_shape[1]
            # precompute all actions for this turn
            # get allies
            allies = np.array([r for r in list(game.robots.values()) + list(game.zombies.values())
                               if r.player_id == self.player_id])
            # determine which will play randomly
            random_index = np.random.rand(len(allies)) <= self.epsilon

            num_deterministic = 0
            for i, bot in enumerate(allies):
                bot.next_state, bot.reward, bot.done = self.percept(game, bot)
                if random_index[i]:
                    # if it's random put random permutation of actions in priority list
                    self.priority_actions_array[i, :] = np.random.permutation(range(action_size))
                else:
                    # if it's deterministic, put its state into the array
                    self.state_array[num_deterministic, :] = bot.next_state
                    num_deterministic += 1

            # compute values for each action for deterministic bots only
            self.priority_actions_array[:len(allies)][~random_index, :] = np.argsort(
                self.model(self.state_array[:num_deterministic]).numpy(),
                axis=1
            )[:, ::-1]

            # assign valid actions based on the priority lists
            for i, bot in enumerate(allies):
                bot.next_action_index, bot.next_action = self.get_valid_action(
                    self.priority_actions_array[i, :], game, bot
                )

        # get and use extra information from the robot AttrDict stored in game.robots
        if not self.exploit and robot.state is not None:
            self.remember(robot.state, robot.action, robot.reward, robot.next_state, robot.done)
            # exponential moving average
            self.ema_reward += self.ema_alpha * (robot.reward - self.ema_reward)

        if not robot.done:
            # store state and action for next time
            self.state, self.action = robot.next_state, robot.next_action_index
            return robot.next_action
        else:
            # This robot isn't going to do anything, it's done.
            return ['guard']

    def train(self):
        if not self.exploit and len(self.memory) >= self.mini_batch_size:
            mini_batch = random.sample(self.memory, self.mini_batch_size)
            for i, (state, action_index, reward, next_state, done) in enumerate(mini_batch):
                self.state_array[i, :] = state
                self.action_array[i] = action_index
                self.target_array[i] = reward
                self.next_state_array[i, :] = next_state if next_state is not None else np.zeros_like(state)
                self.done_array[i] = done

            target_f = self.model(self.state_array).numpy()
            self.target_array[~self.done_array] = \
                self.target_array[~self.done_array] + \
                self.gamma * np.amax(self.model(self.next_state_array[~self.done_array, :]).numpy())
            target_f[range(len(target_f)), self.action_array] = self.target_array
            self.model.fit(self.state_array, target_f, batch_size=32, epochs=1, verbose=0)

            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

    def save(self):
        if not self.exploit:
            self.model.save(self.model_file)

    def load(self):
        self.model = load_model(os.path.join(self.model_dir, 'model.h5'), compile=True)

    @staticmethod
    @abstractmethod
    def _build_model(**model_params):
        pass

    @staticmethod
    @abstractmethod
    def get_reward(game, robot):
        pass

    @staticmethod
    @abstractmethod
    def get_state(game, robot):
        pass

    @staticmethod
    @abstractmethod
    def get_action(action_index, game, robot):
        pass


def get_logger(model_dir):
    # noinspection SpellCheckingInspection
    fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    log_file = os.path.join(model_dir, 'robot_game.log')
    logger = logging.Logger('robot.logger')
    formatter = logging.Formatter(fmt)
    file_log_handler = logging.FileHandler(log_file)
    file_log_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_log_handler)
    logger.addHandler(stream_handler)
    return logger


def get_player(path):
    if os.path.isdir(path):
        # if it's a model directory
        model_dir = path
        params_file = os.path.join(model_dir, 'params.json')
        assert os.path.isfile(params_file), f'Failed to find {params_file}'

        with open(params_file) as fp:
            model_params = json.load(fp)

        robot_file = os.path.join(model_dir, 'robot_game.py')
        assert os.path.isfile(robot_file), f'Failed to find {robot_file}'

        spec = spec_from_file_location('robot_game', robot_file)
        module = module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        # if it's a robot file
        import importlib
        module = importlib.import_module(path)
        model_params = {}
    robot = getattr(module, 'Robot')(**model_params)
    return rg_game.Player(robot=robot), robot


if __name__ == '__main__':
    player, robot = get_player('drl_robot/20211104224501')
    print(player, robot)
