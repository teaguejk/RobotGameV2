from rgkit import rg

# from geo.astar_v1 import AStar
# from geo.astar_v2 import AStar
# from geo.data_structs import Node
# from geo.board import Board

class Cell(object):

    def __init__(self, x: int, y: int, distance: float):
        self.x = x
        self.y = y
        self.distance = distance
        self.gradient = None

    def __eq__(self, other):
        if isinstance(other, Cell):
            return self.x == other.x and self.y == other.y
        return False

    def __hash__(self):
        return hash((self.x, self.y))

    # ------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def distance(a, b):
        """
        Distance between two cells
        @param a: Cell
        @param b: Cell
        @return:
        """
        return rg.dist((a.x, a.y), (b.x, b.y))

    # ------------------------------------------------------------------------------------------------------------------

    @property
    def weight(self):
        if self.gradient is not None:
            return self.gradient + self.distance
        return None

# ----------------------------------------------------------------------------------------------------------------------


class AStar(object):

    def __init__(self, board):
        self.board = board

    # ------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def __neighbors(weighed_board: list, cell: Cell):
        neighbors = []
        if cell.x - 1 >= 0:
            n = weighed_board[cell.x - 1][cell.y]
            if n:
                neighbors.append(n)
        if cell.x + 1 < rg.settings.board_size:
            n = weighed_board[cell.x + 1][cell.y]
            if n:
                neighbors.append(n)
        if cell.y - 1 >= 0:
            n = weighed_board[cell.x][cell.y - 1]
            if n:
                neighbors.append(n)
        if cell.y + 1 < rg.settings.board_size:
            n = weighed_board[cell.x][cell.y + 1]
            if n:
                neighbors.append(n)
        return neighbors

    @staticmethod
    def __reconstruct_path(came_from: dict, current: list):
        """
        Constructs the path to `current` by finding the preceding point of the current one
        :param came_from:
        :param current:
        :return:
        """
        # https://en.wikipedia.org/wiki/A*_search_algorithm
        total_path = [current]
        while current in came_from.keys():
            current = came_from[current]
            total_path.insert(0, current)
        return total_path

    # ------------------------------------------------------------------------------------------------------------------

    def __format_board(self, destination: tuple):
        # TODO: update no-go tiles if needed
        distance = rg.dist
        weighed_board = [ # or tile == 't' # removed team bumping
                          # or tile == 's' # removed for easier movement from start
            [None if (tile == 'o' or tile == 'e') else
             Cell(x, y, distance((x, y), destination)) for y, tile in enumerate(row)] for x, row in enumerate(self.board)]
        return weighed_board

    def __navigator(self, start: Cell, destination, neighbors):
        # https://en.wikipedia.org/wiki/A*_search_algorithm
        start.gradient = 0
        open_set = [start]
        closed_set = []
        came_from = {}

        while len(open_set) > 0:
            current = min(open_set, key=lambda open_set: open_set.weight)
            if current == destination:
                return self.__reconstruct_path(came_from, current)

            open_set.remove(current)
            closed_set.append(current)
            for neighbor in neighbors(current):
                if neighbor in closed_set:
                    continue
                g = current.gradient + Cell.distance(current, neighbor)
                if neighbor.gradient is None or g < neighbor.gradient:
                    came_from[neighbor] = current
                    neighbor.gradient = g
                    if neighbor not in open_set:
                        open_set.append(neighbor)
        return None

    # ------------------------------------------------------------------------------------------------------------------

    def compute(self, start=(0,0), destination=(0,0)):
        weighed_board = self.__format_board(destination)
        start_cell = weighed_board[start[0]][start[1]]
        end_cell = weighed_board[destination[0]][destination[1]]
        cells = self.__navigator(start_cell, end_cell, lambda x: self.__neighbors(weighed_board, x))
        path = []
        if cells is not None:
            for cell in cells:
                path.append((cell.x, cell.y))
        return path



class Node:
    def __init__(self, location=(0, 0), traversal_cost=0):
        self.location = location
        self.f_score = 0
        self.g_score = 0
        self.h_score = traversal_cost
        # self.connected_nodes = []

    def __eq__(self, other):
        if isinstance(other, Node):
            fact = self.location == other.location
            return fact
        return False

    def __hash__(self):
        # + self.g_score + self.h_score # + len(self.connected_nodes)
        return self.location[0] + self.location[1]

    def __str__(self):
        return f"{self.location}: {self.get_f()}"# + ", g: " + str(self.g_score) + ", h: " + str(self.h_score) + ", f: " + str(self.f_score)

    @staticmethod
    def connected_nodes(location):
        nodes = []
        for n in rg.locs_around(location, filter_out=('invalid', 'obstacle')):
            if n is not None:
                nodes.append(Node((n[0], n[1]), 0))
        return nodes

    def reset_node_scores(self):
        self.g_score = 0
        self.h_score = 0

    def get_f(self):
        return self.g_score + self.h_score

    def get_g(self):
        return self.g_score

    def get_h(self):
        return self.h_score / 3

    def create_valid_neighbors(self, goal_node, game):
        target = goal_node
        neighbors = PriorityQueue()
        for node in Node.connected_nodes(self.location):
            node.g_score = self.g_score + rg.dist(self.location, node.location)
            node.h_score = rg.dist(node.location, target.location)
            if node.location in game.robots:
                node.h_score = node.h_score + 5
            neighbors.add(node, -node.get_f())
        return neighbors

    def compare_to(self, other):
        if isinstance(other, Node):
            if self.get_f() == other.get_f():
                return 0
            else:
                return self.get_f() > other.get_f()
        return 1


# By Fabian Flechtmann: flechtmann.net
# https://github.com/fafl/priority-queue/blob/master/priority_queue.py
# Modified by Ethan Gunter
class PriorityQueue(object):

    def __init__(self):

        # List of items, flattened binary heap. The first element is not used.
        # Each node is a tuple of (value, priority, insert_counter)
        self.nodes = [None]  # first element is not used

        # Current state of the insert counter
        self.insert_counter = 0          # tie breaker, keeps the insertion order

    # Comparison function between two nodes
    # Higher priority wins
    # On equal priority: Lower insert counter wins
    def _is_higher_than(self, a, b):
        return b[1] < a[1] or (a[1] == b[1] and a[2] < b[2])

    # Move a node up until the parent is bigger
    def _heapify(self, new_node_index):
        while 1 < new_node_index:
            new_node = self.nodes[new_node_index]
            parent_index = int(new_node_index / 2)
            parent_node = self.nodes[parent_index]

            # Parent too big?
            if self._is_higher_than(parent_node, new_node):
                break

            # Swap with parent
            tmp_node = parent_node
            self.nodes[parent_index] = new_node
            self.nodes[new_node_index] = tmp_node

            # Continue further up
            new_node_index = parent_index

    # Add a new node with a given priority
    def add(self, value, priority):
        new_node_index = len(self.nodes)
        self.insert_counter += 1
        self.nodes.append((value, priority, self.insert_counter))

        # Move the new node up in the hierarchy
        self._heapify(new_node_index)

    # Return the top element
    def peek(self):
        if len(self.nodes) == 1:
            return None
        else:
            return self.nodes[1][0]

    # Remove the top element and return it
    def pop(self):

        if len(self.nodes) == 1:
            raise LookupError("Heap is empty")

        result = self.nodes[1][0]

        # Move empty space down
        empty_space_index = 1
        while empty_space_index * 2 < len(self.nodes):

            left_child_index = empty_space_index * 2
            right_child_index = empty_space_index * 2 + 1

            # Left child wins
            if (
                len(self.nodes) <= right_child_index
                or self._is_higher_than(self.nodes[left_child_index], self.nodes[right_child_index])
            ):
                self.nodes[empty_space_index] = self.nodes[left_child_index]
                empty_space_index = left_child_index

            # Right child wins
            else:
                self.nodes[empty_space_index] = self.nodes[right_child_index]
                empty_space_index = right_child_index

        # Swap empty space with the last element and heapify
        last_node_index = len(self.nodes) - 1
        self.nodes[empty_space_index] = self.nodes[last_node_index]
        self._heapify(empty_space_index)

        # Throw out the last element
        self.nodes.pop()

        return result

    def __contains__(self, item):
        for it in self.nodes:
            if it is None:
                continue
            if item in it:
                return True
        return False


class AStar(object):

    @staticmethod
    def find_path(start_node: Node, goal_node: Node, game, max_search_time=10):
        back_pointers = {}
        closed_set = set()
        open_set = PriorityQueue()
        time_start = 0
        open_set.add(start_node, -start_node.get_f())
        start_node.reset_node_scores()
        time = 0

        while open_set.peek() is not None:
            current = open_set.pop()
            time += 1
            if time - time_start > max_search_time:
                return AStar._reconstruct(current, back_pointers)
            if current == goal_node:
                return AStar._reconstruct(current, back_pointers)

            closed_set.add(current)
            nbrs = current.create_valid_neighbors(goal_node, game)
            while nbrs.peek() is not None:
                nbr = nbrs.pop()
                if nbr is not None:
                    if nbr in closed_set:
                        continue
                    projected_g = current.g_score + nbr.g_score

                    if not open_set.__contains__(nbr):
                        open_set.add(nbr, -nbr.get_f())
                    elif projected_g >= nbr.g_score:
                        continue

                    # record it
                    back_pointers[nbr] = current
        print("AStar fell through")
        return None

    @staticmethod
    def _reconstruct(current, back_pointers):
        # the back_pointers ref from hash dict where node __hash__ is the key
        path = list()
        try:
            while current in back_pointers.keys():
                path.append(current)
                current = back_pointers[current]
            return path
        except MemoryError:
            path_string = ""
            for i in range(30):
                path_string += path.pop() + " => "
            print("AStar Path out of memory. Check for infinite loops...\n" + str(path_string))

        return None


class Robot:

    def act(self, game):

        start = Node(self.location, 0)

        # if bot is in respawn, get out be moving toward center point
        if 'spawn' in rg.loc_types(self.location):
            return ['move', rg.toward(self.location, rg.CENTER_POINT)]

        closest_enemy = self.closest_enemy(game)
        closest_ally = self.closest_enemy(game)

        # bots near an enemy with <= 5 HP will suicide
        if self.hp <= 5 and rg.dist(closest_enemy, self.location) <= 1:
            return ['suicide']

        all_nodes = [(9, 9), (8, 8), (8, 10), (10, 10), (10, 8), (7, 9), (9, 11), (11, 9), (9, 7), (7, 7),
                                     (7, 11), (11, 11), (11, 7)]

        init_square_formation = [(9, 9), (8, 8), (8, 10), (10, 10), (10, 8)]
        formation = init_square_formation.copy()
        open_formation_spot = self.get_open_formation_spot(game, formation)

        if rg.dist(closest_enemy, self.location) == 1:
            return ['attack', closest_enemy]

        # if bot is in initial formation, and center is open, return guard to help prevent collisions
        if open_formation_spot == rg.CENTER_POINT:
            if (self.location in init_square_formation):
                # return ['guard']
                pass

        # else if, center is open and not on an initial formation square, try to move to center
            elif open_formation_spot == rg.CENTER_POINT:
                goal = Node(open_formation_spot, 0)
                path = AStar.find_path(start, goal, game)
                if path:
                    next_step = path.pop()
                    return ['move', rg.toward(self.location, next_step.location)]

        # bots in initial formation spots will guard if no enemies are nearby
        if (self.location in init_square_formation):
            return ['guard']

        # if enemy is within 2 squares and bot is in formation, it will try to preemptively attack enemies moving close
        for ele in all_nodes:
            if self.location == ele:
                if 1 < rg.wdist(closest_enemy, self.location) <= 2:
                    #  attack adj square
                    return ['attack', rg.toward(self.location, closest_enemy)]
                return ['guard']

        # start checks for open spots in formations
        if open_formation_spot is None:
            lil_diamond_formation = [(7, 9), (9, 11), (11, 9), (9, 7), (7, 7),
                                     (7, 11), (11, 11), (11, 7)]
            formation = lil_diamond_formation.copy()
            open_formation_spot = self.get_open_formation_spot(game, formation)
            if open_formation_spot is None:
                big_diamond_formation = [(8, 6), (6, 8), (6, 10), (8, 12), (10, 12), (12, 10), (12, 8),
                                         (10, 6), (9, 5), (5, 9), (9, 13), (13, 9)]
                formation = big_diamond_formation.copy()
                open_formation_spot = self.get_open_formation_spot(game, formation)

        # attempt to stop colliding for center point
        if rg.dist(closest_ally, self.location) <= 2 and open_formation_spot == rg.CENTER_POINT:
            return ['guard']

        if open_formation_spot == rg.CENTER_POINT:
            return ['move', rg.toward(self.location, open_formation_spot)]

        # if theres an enemy 2 tiles away with no adjacent enemies, preemptively attack adj square
        for ele in formation:
            if self.location == ele:
                if 1 < rg.wdist(closest_enemy, self.location) <= 2:
                    #  attack adj square
                    return ['attack', rg.toward(self.location, closest_enemy)]
                return ['guard']

        # if there are no more spots in any formation
        if open_formation_spot is None:
            return ['move', rg.toward(self.location, closest_enemy)]

        # if theres an open spot in formation, try to move and fill it
        if open_formation_spot:
            goal = Node(open_formation_spot, 0)
            path = AStar.find_path(start, goal, game)
            if path:
                next_step = path.pop()
                return ['move', next_step.location]

            else:
                return ['guard']

# returns location of closest enemy bot
    def closest_enemy(self, game):
        closest_enemy = (1000, 1000)
        for loc, bot in game.get('robots').items():
            if bot.player_id != self.player_id:
                if rg.wdist(loc, self.location) <= rg.wdist(closest_enemy, self.location):
                    closest_enemy = loc

        return closest_enemy

# returns location of closest ally bot
    def closest_ally(self, game):
        closest_ally = (1000, 1000)
        for loc, bot in game.get('robots').items():
            if bot.player_id == self.player_id:
                if rg.wdist(loc, self.location) <= rg.wdist(closest_ally, self.location):
                    closest_ally = loc

        return closest_ally

# checks for, and returns location of open spot in formation
    def get_open_formation_spot(self, game, formation):
        # copy formation so its not overwritten
        li = formation.copy()
        for loc, bot in game.robots.items():    # all bots in game
            if bot.player_id == self.player_id:  # if bot is friendly...
                for spot in li:
                    # if loc of friendly bot is in formation, remove spot from formation list
                    if loc == spot:
                        li.remove(spot)
                        # once we know this specific bot is in formation we don't have to check rest of list
                        break

        # if list is empty, return None. Meaning all spots in formation are filled
        if not li:
            return None
        else:
            return li[0]  # returns first open spot in diamond formation list
