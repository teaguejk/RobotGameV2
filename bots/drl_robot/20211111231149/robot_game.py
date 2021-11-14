import datetime
import json
import os
import shutil
import sys
import time
import numpy as np
import rgkit.rg as rg
from rgkit import game as rg_game
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from rgkit.bots.drl_robot_helpers import DRLRobot, get_player, get_logger


class Robot(DRLRobot):
    def __init__(self, model_dir='.', exploit=True, mini_batch_size=1000, memory_size=10000, epsilon_decay=0.99,
                 **model_params):
        super().__init__(model_dir=model_dir, exploit=exploit, mini_batch_size=mini_batch_size,
                         epsilon_decay=epsilon_decay, memory_size=memory_size, **model_params)

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
        if action_index == 8:
            return ['guard']
        elif action_index == 9:
            return ['suicide']
        else:
            locations = rg.locs_around(robot.location)
            if action_index < 4:
                return ['move', locations[action_index]]
            else:
                return ['attack', locations[action_index - 4]]

    @staticmethod
    def enemy_at_loc(game, robot, loc):
        if loc in game.robots:
            if game.robots[loc].player_id != robot.player_id:
                return True
        return False

    @staticmethod
    def get_state(game, robot):
        """
        Return a numpy 'nd-array' representing this robot's state within the game.

        :param game: The game information
        :param robot: The robot to compute the state for.
        :return: The robot's state as a numpy array
        """

        # state = [on spawn?, spawn turn?, enemy_down?, enemy_right?, enemy_up?, enemy_left?]
        state = [
            'spawn' in rg.loc_types(robot.location),
            game.turn % 10 == 0,
        ] + [
            Robot.enemy_at_loc(game, robot, loc) for loc in rg.locs_around(robot.location)
        ]

        return np.array(state, dtype=np.float32)

    @staticmethod
    def get_reward(game, robot):
        """
        You can use the robot fields in 'self' and the game information to determine what reward to give the robot.

        :param game: the game information
        :param robot: the robot
        :return: a number indicating reward (higher is better)
        """
        if robot.hp <= 0:
            # death
            return -1.0
        elif game.turn == 99:
            # survive
            return 1.0
        else:
            # otherwise
            return 0.0


def main():
    self_play = True
    params = {
        'learning_rate': 0.001,
        'layers': [32, 32],
        'activation': 'relu',
        'mini_batch_size': 1000,  # roughly one game's worth of actions
        'memory_size': 10000,  # roughly 10 games worth of actions
        'reg_const': 0.000,
        'epsilon_decay': 0.99,
        'state_size': (6,),
        'action_size': 10,
    }

    if len(sys.argv) > 1:
        model_dir = sys.argv[1]
    else:
        model_dir = os.path.join('drl_robot', datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S"))

    if len(sys.argv) > 2:
        opponent = sys.argv[2]
    else:
        opponent = 'random_bot'

    if not os.path.isdir(model_dir):
        print(f'Creating {model_dir}')
        os.makedirs(model_dir, exist_ok=True)
        # write params file
        with open(os.path.join(model_dir, 'params.json'), 'w') as fp:
            json.dump(params, fp)
        shutil.copyfile(__file__, os.path.join(model_dir, 'robot_game.py'))
        shutil.copyfile('drl_robot_helpers.py', os.path.join(model_dir, 'robot_game_helpers.py'))
    else:
        with open(os.path.join(model_dir, 'params.json')) as fp:
            params = json.load(fp)

    logger = get_logger(model_dir)

    logger.info(f'{model_dir} vs. {opponent}, self_play={self_play}')

    robot1 = Robot(model_dir=model_dir, exploit=False, **params)
    player1 = rg_game.Player(robot=robot1)

    if self_play:
        player2 = rg_game.Player(robot=robot1)
        player3, robot3 = get_player(opponent)
    else:
        player2, robot2 = get_player(opponent)
        player3, robot3 = None, None

    check_states = np.array([
        [0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
    ], dtype=np.float32)

    logger.info('\n' + str(robot1.model(check_states).numpy().round(2)))

    average_score = 0
    num_episodes = 1000  # number of games to train
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
        average_score += 0.01 * (score - average_score)

        # log the results
        logger.info(f'episode: {e}/{num_episodes}, score = {scores[0]:2d} - {scores[1]:2d} = {score:2d}, '
                    f'e: {robot1.epsilon:.3f}, average_score: {average_score:.2f}, '
                    f'average_reward: {robot1.ema_reward:.3f}, '
                    f'play: {t_play - t0:.1f} s., train: {t_train - t_play:.1f} s.')

        # save the model every 50 games.
        if e % 50 == 0:
            robot1.save()
            # log the expected future reward for actions in two states
            logger.info('\n' + str(robot1.model(check_states).numpy().round(2)))
            if self_play:
                # compare against opponent
                # set robot to exploit model
                robot1.exploit = True
                # play 50 games
                num_games = 10
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
                logger.info(f'vs. {opponent}: {wins}-{loss}-{draw}, average score = {score} - {opp_score} = {opponent_average_score}, '
                            f'{t_opp:.1f} s.')
                robot1.exploit = False

    logger.info(f'{(time.time() - t) / num_episodes:.3f} s. per episode')


if __name__ == '__main__':
    main()
