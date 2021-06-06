"""
In search.py, you will implement generic search algorithms
"""

import util


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        util.raiseNotDefined()

    def is_goal_state(self, state):
        """
        state: Search state

        Returns True if and only if the state is a valid goal state
        """
        util.raiseNotDefined()

    def get_successors(self, state):
        """
        state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        util.raiseNotDefined()

    def get_cost_of_actions(self, actions):
        """
        actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        util.raiseNotDefined()


def depth_first_search(problem: SearchProblem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches
    the goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:
    """
    "*** YOUR CODE HERE ***"
    cur_state = problem.get_start_state()
    visited_states = set()
    actions = util.Stack()
    stack_sol = dfs_helper(problem, cur_state, visited_states, actions)
    sol_list = []
    while not stack_sol.isEmpty():
        item = stack_sol.pop()
        sol_list.insert(0, item)
    return sol_list


def dfs_helper(problem, cur_state, visited_states, actions):
    if problem.is_goal_state(cur_state):
        return actions
    visited_states.add(cur_state)
    for (successor, action, stepCost) in problem.get_successors(cur_state):
        if successor not in visited_states:
            actions.push(action)
            # return path if found a successful one,
            # else: continue to other successors
            partial_path = dfs_helper(problem, successor,
                                      visited_states, actions)
            if partial_path is not None:
                return partial_path
            # didn't fount goal in this successor
            actions.pop()


def get_path_from_dict(dict, goal_state_index):
    IDX = 0
    ACTION = 1

    sol_list = []
    cur_index = goal_state_index
    while dict[cur_index][IDX] is not None:
        sol_list.insert(0, dict[cur_index][ACTION])
        cur_index = dict[cur_index][IDX]
    return sol_list


def breadth_first_search(problem):
    """
    Search the shallowest nodes in the search tree first.
    """
    # cur state here is the first state
    cur_state = problem.get_start_state()
    states_queue = util.Queue()
    states_queue.push(cur_state)
    visited_states = set()
    visited_states.add(cur_state)
    # create dictionary for prev states
    # index - give states a number according too the order in the Queue
    prev_dict = dict()  # dict like -> {index:(prev_index, state)}
    prev_dict[0] = (None, cur_state)  # no previous state to the first state.
    index = 1
    cur_state_index = -1  # for += 1 = 0

    if problem.is_goal_state(cur_state):
        return []

    while not states_queue.isEmpty():
        cur_state = states_queue.pop()
        cur_state_index += 1
        for (successor, action, stepCost) in problem.get_successors(cur_state):
            if successor not in visited_states:
                # add state to previous dictionary
                prev_dict[index] = (cur_state_index, action)
                index += 1
                # check if goal state
                if problem.is_goal_state(successor):
                    return get_path_from_dict(prev_dict, index - 1)

                states_queue.push(successor)  # add to the Queue of states to discover
                visited_states.add(successor)  # mark state as visited


def uniform_cost_search(problem):
    """
    Search the node of least total cost first.
    """
    first_state = problem.get_start_state()
    visited = set()
    actions = []
    board_action_dict = dict()
    priority_queue = util.PriorityQueue()
    # item in priority_queue = (state(np.array), move/action, actions), cost
    priority_queue.push(first_state, 0)  # (first_state, None, actions), 0)  # cost = 0
    board_action_dict[first_state] = actions
    while not priority_queue.isEmpty():
        board = priority_queue.pop()
        if problem.is_goal_state(board):
            return board_action_dict[board]  # return the board's actions list
        if board not in visited:
            visited.add(board)
            for (successor, action, stepCost) in problem.get_successors(board):
                if successor not in visited:
                    successor_moves = list(board_action_dict[board])
                    successor_moves += [action]
                    board_action_dict[successor] = successor_moves
                    priority_queue.push(successor, problem.get_cost_of_actions(board_action_dict[successor]))


def null_heuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def a_star_search(problem, heuristic=null_heuristic):
    """
    Search the node that has the lowest combined cost and heuristic first.
    """
    ACTIONS = 0
    COST = 1

    first_state = problem.get_start_state()
    visited = set()
    actions = []
    board_action_dict = dict()
    priority_queue = util.PriorityQueue()
    # item in priority_queue = (state(np.array), move/action, actions), cost
    priority_queue.push(first_state, 0)  # (first_state, None, actions), 0)  # cost = 0
    board_action_dict[first_state] = (actions, 0)
    while not priority_queue.isEmpty():
        board = priority_queue.pop()
        if problem.is_goal_state(board):
            return board_action_dict[board][ACTIONS]  # return the board's actions list
        if board not in visited:
            visited.add(board)
            for (successor, action, stepCost) in problem.get_successors(board):
                successor_moves = list(board_action_dict[board][ACTIONS])
                successor_moves += [action]
                cost = problem.get_cost_of_actions(successor_moves)
                if successor not in visited or cost < board_action_dict[successor][COST]:
                    board_action_dict[successor] = (successor_moves, cost)
                    priority_with_heuristic = cost + heuristic(successor, problem)
                    priority_queue.push(successor, priority_with_heuristic)


# Abbreviations
bfs = breadth_first_search
dfs = depth_first_search
astar = a_star_search
ucs = uniform_cost_search
