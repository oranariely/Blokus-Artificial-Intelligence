from board import Board
from search import SearchProblem, ucs
import util
import numpy as np
from search import *


class BlokusFillProblem(SearchProblem):
    """
    A one-player Blokus game as a search problem.
    This problem is implemented for you. You should NOT change it!
    """

    def __init__(self, board_w, board_h, piece_list, starting_point=(0, 0)):
        self.board = Board(board_w, board_h, 1, piece_list, starting_point)
        self.expanded = 0

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        return self.board

    def is_goal_state(self, state):
        """
        state: Search state
        Returns True if and only if the state is a valid goal state
        """
        return not any(state.pieces[0])

    def get_successors(self, state):
        """
        state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        # Note that for the search problem, there is only one player - #0
        self.expanded = self.expanded + 1
        return [(state.do_move(0, move), move, 1) for move in state.get_legal_moves(0)]

    def get_cost_of_actions(self, actions):
        """
        actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        return len(actions)


class BlokusCornersProblem(SearchProblem):
    def __init__(self, board_w, board_h, piece_list, starting_point=(0, 0)):
        self.targets = [(board_w - 1, board_h - 1), (board_w - 1, 0), (0, board_h - 1)]
        self.expanded = 0
        self.board_w = board_w
        self.board_h = board_h
        self.board = Board(board_w, board_h, 1, piece_list, starting_point)

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        return self.board

    def is_goal_state(self, state):
        return state.state[0, state.board_w - 1] != -1 and \
               state.state[state.board_h - 1, state.board_w - 1] != -1 and \
               state.state[0, 0] != -1 and \
               state.state[state.board_h - 1, 0] != -1

    def get_successors(self, state):
        """
        state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        # Note that for the search problem, there is only one player - #0
        self.expanded = self.expanded + 1
        return [(state.do_move(0, move), move, move.piece.get_num_tiles()) \
                for move in state.get_legal_moves(0)]

    def get_cost_of_actions(self, actions):
        """
        actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves
        """
        sum_costs = 0
        for action in actions:
            sum_costs += action.piece.get_num_tiles()

        return sum_costs


def find_dist(board_h, board_w, board_as_2d_Array):
    """
    return the distance from the (0,0) point of the closest point that is marked
    """
    k = 0
    while k < max(board_w, board_h):
        if k < board_h:
            for i in range(0, min(board_h - 1, k + 1)):
                if board_as_2d_Array[i, k] != -1:
                    return k
        if k < board_w:
            for j in range(0, min(board_w - 1, k)):
                if board_as_2d_Array[k, j] != -1:
                    return k
        k += 1


def check_near_target(problem, state):
    w = problem.board_w - 1  # y
    h = problem.board_h - 1  # x
    for target in problem.targets:
        x = target[0]
        y = target[1]
        if state.state[x, y] == -1:  # if the target is empty, check it's not blocked.
            if x < h:
                if state.state[x + 1, y] != -1:  # not empty
                    return np.inf
            if x > 0:
                if state.state[x - 1, y] != -1:  # not empty
                    return np.inf
            if y < w:
                if state.state[x, y + 1] != -1:  # not empty
                    return np.inf
            if y > 0:
                if state.state[x, y - 1] != -1:  # not empty
                    return np.inf
    return 0


def blokus_corners_heuristic(state, problem):
    """
    Your heuristic for the BlokusCornersProblem goes here.

    This heuristic must be consistent to ensure correctness.  First, try to come up
    with an admissible heuristic; almost all admissible heuristics will be consistent
    as well.

    If using A* ever finds a solution that is worse uniform cost search finds,
    your heuristic is *not* consistent, and probably not admissible!  On the other hand,
    inadmissible or inconsistent heuristics may find optimal solutions, so be careful.
    """
    w = state.board_w
    h = state.board_h
    state_mat = state.state
    if check_near_target(problem, state) != 0:  # it's infinity
        return np.inf
    return find_dist(h, w, state_mat) + find_dist(h, w, np.flip(state_mat, 0)) \
           + find_dist(h, w, np.flip(state_mat, 1)) + find_dist(h, w, np.flip(np.flip(state_mat, 0), 1))


class BlokusCoverProblem(SearchProblem):
    def __init__(self, board_w, board_h, piece_list, starting_point=(0, 0), targets=[(0, 0)]):
        self.targets = targets.copy()
        self.expanded = 0
        self.board_w = board_w
        self.board_h = board_h
        self.board = Board(board_w, board_h, 1, piece_list, starting_point)

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        return self.board

    def is_goal_state(self, state):
        for point in self.targets:
            if state.state[point[0], point[1]] == -1:
                return False
        return True

    def get_successors(self, state):
        """
        state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        # Note that for the search problem, there is only one player - #0
        self.expanded = self.expanded + 1
        return [(state.do_move(0, move), move, move.piece.get_num_tiles()) for move in state.get_legal_moves(0)]

    def get_cost_of_actions(self, actions):
        """
        actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        sum_costs = 0
        for action in actions:
            sum_costs += action.piece.get_num_tiles()

        return sum_costs


def find_dist2(board_h, board_w, board_as_2d_Array, point):
    """
    return the distance from the (0,0) point of the closest point that is marked
    """
    x = point[0]
    y = point[1]
    k = 0
    while k < max(board_w, board_h) or (not (x - k < 0 and y - k < 0)):
        # row: x + k
        if x + k < board_h:
            for j in range(max(0, y - k), min(board_w - 1, y + k + 1)):
                if board_as_2d_Array[x + k, j] != -1:
                    return k
        # column: y + k
        if y + k < board_w:
            for i in range(max(0, x - k), min(board_h - 1, x + k + 1)):
                if board_as_2d_Array[i, y + k] != -1:
                    return k
        # row: x - k
        if x - k >= 0:
            for j in range(max(0, y - k), min(board_h - 1, y + k + 1)):
                if board_as_2d_Array[x - k, j] != -1:
                    return k
        # column: y - k
        if y - k >= 0:
            for i in range(max(0, x - k), min(board_w - 1, x + k + 1)):
                if board_as_2d_Array[i, y - k] != -1:
                    return k
        k += 1


def blokus_cover_heuristic(state, problem):
    sum_heuristics = 0
    if check_near_target(problem, state) != 0:  # it's infinity
        return np.inf
    for point in problem.targets:
        sum_heuristics += find_dist2(state.board_h, state.board_w, state.state, point)
    return sum_heuristics


class ClosestLocationSearch:
    """
    In this problem you have to cover all given positions on the board,
    but the objective is speed, not optimality.
    """

    def __init__(self, board_w, board_h, piece_list, starting_point=(0, 0), targets=(0, 0)):
        self.expanded = 0
        self.start_point = starting_point
        self.targets = targets.copy()
        self.board = Board(board_w, board_h, 1, piece_list, starting_point)

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        return self.board

    def solve(self):
        """
        This method should return a sequence of actions that covers all target locations on the board.
        This time we trade optimality for speed.
        Therefore, your agent should try and cover one target location at a time. Each time, aiming for the closest uncovered location.
        You may define helpful functions as you wish.
        """
        current_state = self.board.__copy__()
        backtrace = []
        problem = BlokusCoverProblem(current_state.board_w, current_state.board_h, current_state.piece_list, self.start_point, self.targets)
        for target in self.targets:
            # set of actions that covers the closets uncovered target location
            problem.board = current_state
            problem.targets = [target]
            actions = bfs(problem)
            # if actions is not None:
            for action in actions:
                current_state.add_move(0, action)
            backtrace += actions

        self.expanded = problem.expanded
        return backtrace



