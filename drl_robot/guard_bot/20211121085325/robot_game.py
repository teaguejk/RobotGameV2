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

        model.add(Dense(1024, activation='relu', kernel_regularizer=l2(reg_const)))
        # model.add(Dense(512, activation='relu', kernel_regularizer=l2(reg_const)))
        model.add(Dense(512, activation='relu', kernel_regularizer=l2(reg_const)))
        for units in layers:
            model.add(Dense(units, activation=activation, kernel_regularizer=l2(reg_const)))
        model.add(Dense(action_size, activation='linear', kernel_regularizer=l2(reg_const)))
        model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))
        return model

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

    @staticmethod
    def friendly_neighborhood(game, robot):
        return [float(loc in game.robots and game.robots[loc].player_id == robot.player_id)
                for loc in rg.locs_around(robot.location)]

    @staticmethod
    def enemy_at_loc(game, robot, loc):
        if loc in game.robots:
            if game.robots[loc].player_id != robot.player_id:
                return True
        return False

    @staticmethod
    def check_if_enemy_is_2away(game, robot):
        if 'spawn' in rg.loc_types(robot.location):
            return False
        for loc in rg.locs_2away(robot.location):
            if Robot.enemy_at_loc(game, robot, loc):
                return True
        return False

    @staticmethod
    def enemy_camping_spawn(game, robot):
        if 'spawn' in rg.loc_types(robot.location) \
                and game.robot[rg.toward(robot.location, rg.CENTER_POINT)].player_id != robot.player_id:
            return True
        return False

    @staticmethod
    def robot_health_above50(game, robot):
        if robot.hp >= 50:
            return True
        return False

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

        # not working yet...
        # if Robot.enemy_camping_spawn(game, robot):
        #     return ['attack', rg.toward(robot.location, rg.CENTER_POINT)]

        if 'spawn' in rg.loc_types(robot.location):
            return ['move', rg.toward(robot.location, rg.CENTER_POINT)]

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
            # Robot.suicide,
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
            Robot.robot_health_above50,
            Robot.enemy_neighborhood,
            Robot.friendly_neighborhood,
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
        # Actions:
        # [0 move up, 1 move right, 2 move down, 3 move left,
        # 4 attack up, 5 attack right, 6 attack down, 7 attack left,
        # 8 guard,
        # 9 suicide]

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

        # print("Action: " + str(robot.action) + ", Turn: " + str(game.turn))
        # if robot.hp <= 0:
        #     print("Died turn: " + str(game.turn)
        #           + ", Last Action: " + str(robot.action)
        #           + ", Location: " + str(robot.location))

        # if game.turn % 10 == 0 and 'spawn' in rg.loc_types(robot.location):
        #     total_reward += -50

        # if robot.action in attack and robot.damage_caused > 0:
        #     print(robot.player_id, ": hit something for " + str(robot.damage_caused))

        # if dead, neg reward
        # if robot.hp <= 0:
        #     return -10

        # if alive at end of game, pos reward
        # if game.turn == 99:
        #     return 10.0
        #
        # # if robot attacks and is above 50 hp and deals damage, pos reward
        # if robot.action in attack and robot.hp >= 50 and adj_enemy and robot.damage_caused > 0:
        #     total_reward += 5.0
        #
        # # if robot moves while below 50 hp, pos reward
        # if robot.action in move and robot.hp < 50:
        #     total_reward += 6.0
        #
        # if game.turn % 10 == 0 and 'spawn' in rg.loc_types(robot.location):
        #     print(robot.player_id, ": I'm in spawn like a dummy on turn: " + str(game.turn))
        #
        # # if robot is in spawn, neg reward
        # if 'spawn' in rg.loc_types(robot.location) and game.turn % 10 == 0:
        #     # print(robot.player_id, ": I'm in spawn like a dummy. Turn: " + str(game.turn))
        #     total_reward += -8.0
        # else:
        #     total_reward += 1.0
        #
        # # if robot attacks with enemy 2 away and no enemies are adjacent, pos reward
        # if 'spawn' not in rg.loc_types(robot.location):
        #     enemy_2away = Robot.check_if_enemy_is_2away(game, robot)
        #     if robot.action in attack and enemy_2away and not adj_enemy:
        #         total_reward += 5
        #
        # # if loc is <= 5 spaces from center, pos reward
        # if rg.wdist(robot.location, rg.CENTER_POINT) <= 6:
        #     total_reward += 2.0
        #
        # # if loc is > 5 spaces from center, pos reward
        # if rg.wdist(robot.location, rg.CENTER_POINT) > 6:
        #     total_reward += -2.0
        #
        # # suicide is not the answer, neg reward
        if robot.action == 9:
            print("Suicided on turn " + str(game.turn) + " for " + str(robot.damage_caused) + " damage.")
            reward += -50.0
        #
        # # if robot takes damage without dealing damage, neg reward
        # if robot.damage_taken >= 0 and robot.damage_caused == 0:
        #     total_reward += -2.0
        #
        # if robot.damage_caused > 0:
        #     total_reward += 5.0

        # if robot.hp != 0:
        #     total_reward += 3.0
        # print(robot.damage_caused)
        # # if suicide and successful, pos reward, else suicide, neg reward
        # if (robot.hp <= 10) and robot.action == 9 and Robot.surrounders(robot, game, robot.location):
        #     # successful suicide
        #     total_reward += 10.0
        # elif (robot.hp <= 10) and robot.action == 9:
        #     total_reward += -20.0
        #
        # # if surrounded and guard, pos reward
        # if robot.action == 8 and Robot.surrounders(robot, game, robot.location):
        #     # successful guard
        #     total_reward += 8.0
        # elif robot.action == 8:
        #     total_reward += -10.0

        # print(total_reward)

        return reward


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
        'layers': [64, 64],
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
        opponent = 'move_to_center_bot'

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
