import random
from rgkit import rg


class Robot:
    def act(self, game):
        locs_around = rg.locs_around(self.location, filter_out=('obstacle', 'invalid'))

        actions = [[a, loc] for a in ['move', 'attack'] for loc in locs_around] + [['guard'], ['suicide']]
        return random.choice(actions)
