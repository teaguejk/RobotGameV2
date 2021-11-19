import datetime
import json
import os
import shutil
import sys
import time
import numpy as np
import random
import rgkit.rg as rg
from rgkit import game as rg_game
from rgkit.gamestate import AttrDict
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from drl_robot_helpers import DRLRobot, get_player, get_logger


class Robot(DRLRobot):
    def __init__(self, model_dir='.', exploit=True, mini_batch_size=1000, memory_size=10000, epsilon_decay=0.99,
                 **model_params):
        super().__init__(model_dir=model_dir, exploit=exploit, mini_batch_size=mini_batch_size,
                         epsilon_decay=epsilon_decay, memory_size=memory_size, **model_params)

#-----------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def _build_model(state_size=(1,), action_size=10, learning_rate=0.001, layers=(32, 32), activation='relu',
                     reg_const=0):
        """
        Build a keras model that takes the game state as input and produces the expected future reward corresponding
        to each possible action.

        :return: a keras model
        """
        model = Sequential()
        model.add(Input(shape=state_size))
        for units in layers:
            model.add(Dense(units, activation=activation, kernel_regularizer=l2(reg_const)))
        model.add(Dense(action_size, activation='linear', kernel_regularizer=l2(reg_const)))
        model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))
        return model

#-----------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def guard(game, robot):
        return ['guard']

    @staticmethod
    def suicide(game, robot):
        return ['suicide']

    @staticmethod
    def move(index):
        def f(game, robot):
            locs = rg.locs_around(robot.location)
            return ['move', locs[index]]
        return f

    @staticmethod
    def attack(index):
        def f(game, robot):
            locs = rg.locs_around(robot.location)
            return ['attack', locs[index]]
        return f

    def move_off_spawn(self, game, robot):
        if 'spawn' in rg.loc_types(robot.location):
            return ['move', rg.toward(self.location, rg.CENTER_POINT)]

#-----------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def on_spawn(game, robot):
        return 'spawn' in rg.loc_types(robot.location)

    @staticmethod
    def spawn_turn(game, robot):
        return game.turn % 10 == 0

    @staticmethod
    def enemy_neighborhood(game, robot):
        return [float(loc in game.robots and game.robots[loc].player_id != robot.player_id)
                for loc in rg.locs_around(robot.location)]

# ----------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def get_action(action_index, game, robot):
        """
        This function converts an action index into a RobotGame action, one of:
        ['guard'], ['suicide'], ['move', loc], ['attack', loc] where loc is
        north, east, south, or west of robot.location.

        :param action_index: index of action
        :param game: The game information
        :param robot: the robot taking the action
        :return: the RobotGame action
        """
        return [
            Robot.move(0),
            Robot.move(1),
            Robot.move(2),
            Robot.move(3),
            Robot.attack(0),
            Robot.attack(1),
            Robot.attack(2),
            Robot.attack(3),
            Robot.guard,
            Robot.suicide,
            Robot.move_off_spawn,
        ][action_index](game, robot)

    @staticmethod
    def get_state(game, robot):
        """
        Return a numpy 'nd-array' representing this robot's state within the game.

        :param game: The game information
        :param robot: The robot to compute the state for.
        :return: The robot's state as a numpy array
        """
        state_funcs = [
            Robot.on_spawn,
            Robot.spawn_turn,
            Robot.enemy_neighborhood,
        ]

        state = []
        for state_func in state_funcs:
            s = state_func(game, robot)
            if isinstance(s, list):
                state.extend(s)
            else:
                state.append(s)

        return np.array(state, dtype=np.float32)

    @staticmethod
    def get_reward(game, robot):
        """
        You can use the robot fields in 'self' and the game information to determine what reward to give the robot.

        :param game: the game information
        :param robot: the robot
        :return: a number indicating reward (higher is better)
        """
        death_penalty = 50
        reward = 0
        if robot.hp <= 0:
            # death
            reward -= death_penalty
        elif game.turn == 99:
            # survive
            reward += death_penalty

        # kills
        reward += robot.kills * death_penalty

        # damage
        reward += robot.damage_caused - robot.damage_taken
        return reward

#-----------------------------------------------------------------------------------------------------------------------

def get_action_size():
    location = (12, 9)
    robot = AttrDict(location=location, player_id=0, robot_id=0, hp=50)
    game = AttrDict(turn=1, robots=AttrDict({location: robot}))
    num_actions = 0
    while True:
        try:
            Robot.get_action(num_actions, game, robot)
            num_actions += 1
        except Exception:
            break
    return num_actions


def get_state_size():
    location = (12, 9)
    robot = AttrDict(location=location, player_id=0, robot_id=0, hp=50)
    game = AttrDict(turn=1, robots=AttrDict({location: robot}))
    state = Robot.get_state(game, robot)
    return state.shape


def main():
    self_play = False
    params = {
        'learning_rate': 0.001,
        'layers': [16, 16],
        'activation': 'tanh',
        'mini_batch_size': 10000,  # roughly 10 game's worth of actions
        'memory_size': 100000,  # roughly 100 games worth of actions
        'reg_const': 0.000,
        'epsilon_decay': 0.995,
        'state_size': get_state_size(),
        'action_size': get_action_size(),
    }

    if len(sys.argv) > 1:
        opponent = sys.argv[1]
    else:
        opponent = 'dulladob'

    if len(sys.argv) > 2:
        model_dir = sys.argv[2]
    else:
        model_dir = os.path.join('drl_robot', opponent, datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S"))

    if not os.path.isdir(model_dir):
        print(f'Creating {model_dir}')
        os.makedirs(model_dir, exist_ok=True)
        # write params file
        with open(os.path.join(model_dir, 'params.json'), 'w') as fp:
            json.dump(params, fp)
        shutil.copyfile(__file__, os.path.join(model_dir, 'robot_game.py'))
        shutil.copyfile('drl_robot_helpers.py', os.path.join(model_dir, 'robot_game_helpers.py'))

    logger = get_logger(model_dir)
    logger.info(f'{model_dir} vs. {opponent}, self_play={self_play}')

    player1, robot1 = get_player(model_dir)
    robot1.exploit = False

    if self_play:
        player2 = rg_game.Player(robot=robot1)
        player3, robot3 = get_player(opponent)
    else:
        player2, robot2 = get_player(opponent)
        player3, robot3 = player2, robot2

    # make some situations to track
    check_states = []
    robot = AttrDict(location=(12, 12), player_id=0, robot_id=0, hp=50)
    near_death_bot = AttrDict(location=(12, 12), player_id=0, robot_id=0, hp=1)
    neighborhood = rg.locs_around(robot.location)

    # game with one near death robot
    game = AttrDict(turn=1, robots=AttrDict({near_death_bot.location: near_death_bot}))
    check_states.append(Robot.get_state(game, near_death_bot))

    # game with one healthy robot
    game = AttrDict(turn=1, robots=AttrDict({robot.location: robot}))
    check_states.append(Robot.get_state(game, robot))

    # games with one adjacent enemy
    for loc in neighborhood:
        enemy = AttrDict(location=loc, player_id=1, robot_id=0, hp=50)
        game.robots = AttrDict({robot.location: robot, loc: enemy})
        check_states.append(Robot.get_state(game, robot))

    # games with one adjacent friend
    for loc in neighborhood:
        friend = AttrDict(location=loc, player_id=0, robot_id=0, hp=50)
        game.robots = AttrDict({robot.location: robot, loc: friend})
        check_states.append(Robot.get_state(game, robot))

    check_states = np.array(check_states, dtype=np.float32)

    y = robot1.model(check_states).numpy().round(1)
    logger.info('\n' + str(np.concatenate((check_states, y, np.argmax(y, axis=1).reshape((-1, 1))), axis=1)))

    average_score = None
    previous_average = float('-inf')
    patience = 3
    retries = 0
    num_episodes = 10000  # number of games to train
    t = time.time()
    for e in range(1, num_episodes+1):
        t0 = time.time()

        # create new game
        game = rg_game.Game([player1, player2], record_actions=False, record_history=False, print_info=False)

        # run all turns in the game
        game.run_all_turns()

        # get final score
        scores = game.get_scores()
        score = scores[0] - scores[1]
        t_play = time.time()

        # train the robot
        robot1.train()
        t_train = time.time()

        # keep exponential running average of final score
        if average_score:
            average_score += 0.01 * (score - average_score)
        else:
            average_score = score

        # log the results
        logger.info(f'episode: {e}/{num_episodes}, score = {scores[0]:2d} - {scores[1]:2d} = {score:3d}, '
                    f'e: {robot1.epsilon:5.3f}, average_score: {average_score:6.2f}, '
                    f'average_reward: {robot1.ema_reward:7.3f}, '
                    f'play: {t_play - t0:4.1f} s., train: {t_train - t_play:4.1f} s.')

        # save the model every 50 games.
        if e % 50 == 0:
            if average_score <= previous_average:
                retries += 1
                logger.info(f'Did not improve on average score: {previous_average}, retry #{retries}')
            else:
                logger.info(f'Improved on average score: {previous_average} => {average_score}, saving.')
                robot1.save()
                previous_average = average_score
                retries = 0

            # log the expected future reward for actions in two states
            y = robot1.model(check_states).numpy().round(1)
            logger.info('\n' + str(np.concatenate((check_states, y, np.argmax(y, axis=1).reshape((-1, 1))), axis=1)))

            # compare against opponent
            # set robot to exploit model
            robot1.exploit = True
            # play 50 games
            num_games = 20
            scores = []
            t_opp = time.time()
            for _ in range(num_games):
                game = rg_game.Game([player1, player3], record_actions=False, record_history=False, print_info=False)
                game.run_all_turns()
                # get final score
                scores.append(game.get_scores())
            t_opp = time.time() - t_opp
            wins = sum([s[0] > s[1] for s in scores])
            loss = sum([s[0] < s[1] for s in scores])
            draw = sum([s[0] == s[1] for s in scores])
            score = sum(s[0] for s in scores) / len(scores)
            opp_score = sum(s[1] for s in scores) / len(scores)
            opponent_average_score = score - opp_score
            logger.info(f'vs. {opponent}: {wins}-{loss}-{draw}, average score = {score} - {opp_score} = '
                        f'{opponent_average_score}, {t_opp:.1f} s.')
            robot1.exploit = False

            if retries >= patience:
                logger.info(f'Stopping after {retries} retries')
                break

    logger.info(f'{(time.time() - t) / num_episodes:.3f} s. per episode')


if __name__ == '__main__':
    main()
