#Jacob Hitzges
"""
  Search
  ------
  This python file utilizes some class originally developed
  by Kevin Molloy, Peter Norvig, and Russell Stewarts teams.
"""


from utils import PriorityQueue

from collections import defaultdict, deque
import math
# import randommes Madison University - Canvas. ATTENTION: Duo two-factor authentication is now required. for this and many other JMU systems. See here for a ...
# Log In to Canvas
# Canvas by Instructure. Log In. Forgot Password? Enter your ...
# Canvas LMS
# Honors College. Canvas site for students in the Honors College.

# Web results

# Login to Canvas : JMU Libraries
# Login to Canvas. Navigate to canvas.jmu.edu. Note: You must have an activated J
import sys
import bisect
from operator import itemgetter
import time


infinity = float('inf')

# ______________________________________________________________________________


class Problem(object):

    """The abstract class for a formal problem. You should subclass
    this and implement the methods actions and result, and possibly
    __init__, goal_test, and path_cost. Then you will create instances
    of your subclass and solve them with the various search functions."""

    def __init__(self, initial, goal=None):
        """The constructor specifies the initial state, and possibly a goal
        state, if there is a unique goal. Your subclass's constructor can add
        other arguments."""
        self.initial = initial
        self.goal = goal

    def actions(self, state):
        """Return the actions that can be executed in the given
        state. The result would typically be a list, but if there are
        many actions, consider yielding them one at a time in an
        iterator, rather than building them all at once."""
        raise NotImplementedError

    def result(self, state, action):
        """Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state)."""
        raise NotImplementedError

    def goal_test(self, state):
        """Return True if the state is a goal. The default method compares the
        state to self.goal or checks for state in self.goal if it is a
        list, as specified in the constructor. Override this method if
        checking against a single self.goal is not enough."""
        # if isinstance(self.goal, list):
        #     return is_in(state, self.goal)
        # else:
        #     return state == self.goal

    def path_cost(self, c, state1, action, state2):
        """Return the cost of a solution path that arrives at state2 from
        state1 via action, assuming cost c to get up to state1. If the problem
        is such that the path doesn't matter, this function will only look at
        state2.  If the path does matter, it will consider c and maybe state1
        and action. The default method costs 1 for every step in the path."""
        return c + 1

    def value(self, state):
        """For optimization problems, each state has a value.  Hill-climbing
        and related algorithms try to maximize this value."""
        raise NotImplementedError
# ______________________________________________________________________________


class Node:

    """A node in a search tree. Contains a pointer to the parent (the node
    that this is a successor of) and to the actual state for this node. Note
    that if a state is arrived at by two paths, then there are two nodes with
    the same state.  Also includes the action that got us to this state, and
    the total path_cost (also known as g) to reach the node.  Other functions
    may add an f and h value; see best_first_graph_search and astar_search for
    an explanation of how the f and h values are handled. You will not need to
    subclass this class."""

    def __init__(self, state, parent=None, action=None, path_cost=0):
        """Create a search tree Node, derived from a parent by an action."""
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.depth = 0
        if parent:
            self.depth = parent.depth + 1

    def __repr__(self):
        return "<Node {}>".format(self.state)

    def __lt__(self, node):
        return self.state < node.state

    def expand(self, problem):
        """List the nodes reachable in one step from this node."""
        return [self.child_node(problem, action)
                for action in problem.actions(self.state)]

    def child_node(self, problem, action):
        """[Figure 3.10]"""
        next_state = problem.result(self.state, action)
        next_node = Node(next_state, self, action,
                    problem.path_cost(self.path_cost, self.state,
                                      action, next_state))
        return next_node
    
    def solution(self):
        """Return the sequence of actions to go from the root to this node."""
        return [node.action for node in self.path()[1:]]

    def path(self):
        """Return a list of nodes forming the path from the root to this node."""
        node, path_back = self, []
        while node:
            path_back.append(node)
            node = node.parent
        return list(reversed(path_back))

    # We want for a queue of nodes in breadth_first_graph_search or
    # astar_search to have no duplicated states, so we treat nodes
    # with the same state as equal. [Problem: this may not be what you
    # want in other contexts.]

    def __eq__(self, other):
        return isinstance(other, Node) and self.state == other.state

    def __hash__(self):
        return hash(self.state)

# ______________________________________________________________________________


# ______________________________________________________________________________
# Uninformed Search algorithms


def breadth_first_tree_search(problem):
    """Search the shallowest nodes in the search tree first.
        Search through the successors of a problem to find a goal.
        The argument frontier should be an empty queue.
        Repeats infinitely in case of loops. [Figure 3.7]"""

    frontier = deque([Node(problem.initial)])  # FIFO queue
    explored_nodes = 1
    while frontier:
        explored_nodes += 1
        node = frontier.popleft()
        if problem.goal_test(node.state):
            return node, explored_nodes
        frontier.extend(node.expand(problem))
    return None, explored_nodes


def depth_first_tree_search(problem):
    """Search the deepest nodes in the search tree first.
        Search through the successors of a problem to find a goal.
        The argument frontier should be an empty queue.
        Repeats infinitely in case of loops. [Figure 3.7]"""

    frontier = [Node(problem.initial)]  # Stack

    while frontier:
        node = frontier.pop()
        if problem.goal_test(node.state):
            return node
        frontier.extend(node.expand(problem))
    return None


def best_first_graph_search(problem, f):
    """Search the nodes with the lowest f scores first.
    You specify the function f(node) that you want to minimize; for example,
    if f is a heuristic estimate to the goal, then we have greedy best
    first search; if f is node.depth then we have breadth-first search."""
    node = Node(problem.initial)
    frontier = PriorityQueue('min', f)
    frontier.append(node)
    explored = set()
    explored_nodes =  0
    while frontier:
        node = frontier.pop()
        explored_nodes += 1
        if problem.goal_test(node.state):
            return node,explored_nodes
        explored.add(node.state)
        for child in node.expand(problem):
            if child.state not in explored and child not in frontier:
                frontier.append(child)
            elif child in frontier:
                incumbent = frontier[child]
                if f(child) < f(incumbent):
                    del frontier[incumbent]
                    frontier.append(child)
    return None, explored_nodes


def uniform_cost_search(problem):
    """[Figure 3.14]"""
    return best_first_graph_search(problem, lambda node: node.path_cost)


def depth_limited_search(problem, limit=50):
    """[Figure 3.17]"""
    def recursive_dls(node, problem, limit):
        if problem.goal_test(node.state):
            return node
        elif limit == 0:
            return 'cutoff'
        else:
            cutoff_occurred = False
            for child in node.expand(problem):
                result = recursive_dls(child, problem, limit - 1)
                if result == 'cutoff':
                    cutoff_occurred = True
                elif result is not None:
                    return result
            return 'cutoff' if cutoff_occurred else None

    # Body of depth_limited_search:
    return recursive_dls(Node(problem.initial), problem, limit)


def iterative_deepening_search(problem):
    """[Figure 3.18]"""
    for depth in range(sys.maxsize):
        result = depth_limited_search(problem, depth)
        if result != 'cutoff':
            return result


# ______________________________________________________________________________
# Informed (Heuristic) Search


greedy_best_first_graph_search = best_first_graph_search
# Greedy best-first search is accomplished by specifying f(n) = h(n).


def astar_search(problem, h=None):
    """A* search is best-first graph search with f(n) = g(n)+h(n).
    You need to specify the h function when you call astar_search, or
    else in your Problem subclass."""

    return best_first_graph_search(problem, lambda n: n.path_cost + h(n))



class EightPuzzle(Problem):

    """ The problem of sliding tiles numbered from 1 to 8 on a 3x3 board,
    where one of the squares is a blank. A state is represented as a tuple with 9
    elements.  Element 0 contans the number in location 0 in the 3x3 grid (see picture below)

    |-----|-----|-----|
    |  0  |  1  |  2  |
    |  3  |  4  |  5  |
    |  6  |  7  | 8   |
    --------------------


    0 represents the empty square
    So, the board below is encoded as (2,4,3,1,5,6,7,8,0)

    |-----|-----|-----|
    |  2  |  4  |  3  |
    |  1  |  5  | 6   |
    |  7  |  8  |    |
    --------------------
    """
    #done
    def __init__(self, initial, goal=(1, 2, 3, 4, 5, 6, 7, 8, 0)):
        """ Define goal state and initialize a problem """

        self.goal = goal
        Problem.__init__(self, initial, goal)
    #done
    def find_blank_square(self, state):
        """Return the index of the blank square in a given state"""

        return state.index(0)
    #done
    def actions(self, state):
        """ Return the actions that can be executed in the given state.
        The result would be a list, since there are only four possible actions
        in any given state of the environment
        You may need to edit this list, for example, since if you are in the top
        left corner of the board, you can not move left or up.  """
        
        possible_actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']

        blank = self.find_blank_square(state)
        count = 0

        #searches array to find 0
        for x in state:
            
            if x == 0:
                break
            else:
                count = count + 1

        #sets possible actions for zero depending on its position
        if count == 0:
            return ['DOWN','RIGHT']
        elif count == 1:
            return ['DOWN', 'LEFT', 'RIGHT']
        elif count == 2:
            return ['DOWN', 'LEFT']
        elif count == 3:
            return ['UP', 'DOWN', 'RIGHT']
        elif count == 4:
            return ['UP', 'DOWN', 'LEFT', 'RIGHT']
        elif count == 5:
            return ['UP', 'DOWN', 'LEFT']
        elif count == 6:
            return ['UP','RIGHT']
        elif count == 7:
            return ['UP','LEFT', 'RIGHT']
        elif count == 8:
            return ['UP','LEFT']
        else:
            return "Failed Actions"

        #return possible_actions
    #done
    def result(self, state, action):
        """ Given state and action, return a new state that is the result of the action.
        Action is assumed to be a valid action in the state """
        
        #moves depending on action
        returnState = state
        if action == 'LEFT':
            zero = returnState.index(0)
            left = zero - 1
            #changes to list and back to tuple for assigments
            l = list(returnState)
            l[zero], l[left] = l[left], l[zero]
            t = tuple(l)
            
            return t

        #all below are the same as above but with different actions
        if action == 'RIGHT':
            zero = returnState.index(0)
            right = zero + 1
            l = list(returnState)
            l[zero], l[right] = l[right], l[zero]
            t = tuple(l)
            
            return t

        if action == 'UP':
            zero = returnState.index(0)
            right = zero - 3
            l = list(returnState)
            l[zero], l[right] = l[right], l[zero]
            t = tuple(l)
            
            return t

        if action == 'DOWN':
            zero = returnState.index(0)
            right = zero + 3
            l = list(returnState)
            l[zero], l[right] = l[right], l[zero]
            t = tuple(l)
            
            return t

        
        
        return returnState
        
    #done
    def goal_test(self, state):
        """ Given a state, return True if state is a goal state or False, otherwise """

        return state == self.goal
    #done
    def check_solvability(self, state):
        """ Checks if the given state is solvable """

        inversion = 0
        for i in range(len(state)):
            for j in range(i, len(state)):
                if state[i] > state[j] != 0:
                    inversion += 1
        
        return inversion % 2 == 0
    
    def h(self, node):

        """ Return the heuristic value for a given state.
        You should write a heuristic function that is as follows:
            h(n) = number of misplaced tiles
        """

        #raise NotImplementedError;

        #__lt__(self, node)
        state = node.state

        #loops through array to find tiles in the wrong spot
        numCheck = 1
        countWrong = 0
        for x in state:
            if numCheck != x:
                countWrong = countWrong + 1
            numCheck = numCheck + 1

        return countWrong - 1
        

    def h2(self, node):
        """ Return the heuristic value for a given state.
        You should write a heuristic function that is as follows:
        h2(n) = manhatten distance to move tiles into place
        """
        #raise NotImplementedError

        #goes through and sets starting coordinate based on the index 
        sum = 0
        for x in node.state:
            if x == 1:
                state = node.state
                if state.index(1) == 0:
                    x1 = 1
                    y1 = 1
                if state.index(1) == 1:
                    x1 = 2
                    y1 = 1
                if state.index(1) == 2:
                    x1 = 3
                    y1 = 1
                if state.index(1) == 3:
                    x1 = 1
                    y1 = 2
                if state.index(1) == 4:
                    x1 = 2
                    y1 = 2
                if state.index(1) == 5:
                    x1 = 3
                    y1 = 2
                if state.index(1) == 6:
                    x1 = 1
                    y1 = 3
                if state.index(1) == 7:
                    x1 = 2
                    y1 = 3
               
                #sets end coordinate for desired point
                x2 = 1
                y2 = 1
                sum = sum + ((abs(x1 - x2)) + (abs(y1 - y2)))
            #repeats process for all
            if x == 2:
                state = node.state
                if state.index(2) == 0:
                    x1 = 1
                    y1 = 1
                if state.index(2) == 1:
                    x1 = 2
                    y1 = 1
                if state.index(2) == 2:
                    x1 = 3
                    y1 = 1
                if state.index(2) == 3:
                    x1 = 1
                    y1 = 2
                if state.index(2) == 4:
                    x1 = 2
                    y1 = 2
                if state.index(2) == 5:
                    x1 = 3
                    y1 = 2
                if state.index(2) == 6:
                    x1 = 1
                    y1 = 3
                if state.index(2) == 7:
                    x1 = 2
                    y1 = 3
               
                x2 = 2
                y2 = 1
                sum = sum + ((abs(x1 - x2)) + (abs(y1 - y2)))
            if x == 3:
                state = node.state
                if state.index(3) == 0:
                    x1 = 1
                    y1 = 1
                if state.index(3) == 1:
                    x1 = 2
                    y1 = 1
                if state.index(3) == 2:
                    x1 = 3
                    y1 = 1
                if state.index(3) == 3:
                    x1 = 1
                    y1 = 2
                if state.index(3) == 4:
                    x1 = 2
                    y1 = 2
                if state.index(3) == 5:
                    x1 = 3
                    y1 = 2
                if state.index(3) == 6:
                    x1 = 1
                    y1 = 3
                if state.index(3) == 7:
                    x1 = 2
                    y1 = 3
               
                x2 = 3
                y2 = 1
                sum = sum + ((abs(x1 - x2)) + (abs(y1 - y2)))
            if x == 4:
                state = node.state
                if state.index(4) == 0:
                    x1 = 1
                    y1 = 1
                if state.index(4) == 1:
                    x1 = 2
                    y1 = 1
                if state.index(4) == 2:
                    x1 = 3
                    y1 = 1
                if state.index(4) == 3:
                    x1 = 1
                    y1 = 2
                if state.index(4) == 4:
                    x1 = 2
                    y1 = 2
                if state.index(4) == 5:
                    x1 = 3
                    y1 = 2
                if state.index(4) == 6:
                    x1 = 1
                    y1 = 3
                if state.index(4) == 7:
                    x1 = 2
                    y1 = 3
               
                x2 = 1
                y2 = 2
                sum = sum + ((abs(x1 - x2)) + (abs(y1 - y2)))
            if x == 5:
                state = node.state
                if state.index(5) == 0:
                    x1 = 1
                    y1 = 1
                if state.index(5) == 1:
                    x1 = 2
                    y1 = 1
                if state.index(5) == 2:
                    x1 = 3
                    y1 = 1
                if state.index(5) == 3:
                    x1 = 1
                    y1 = 2
                if state.index(5) == 4:
                    x1 = 2
                    y1 = 2
                if state.index(5) == 5:
                    x1 = 3
                    y1 = 2
                if state.index(5) == 6:
                    x1 = 1
                    y1 = 3
                if state.index(5) == 7:
                    x1 = 2
                    y1 = 3
               
                x2 = 2
                y2 = 2
                sum = sum + ((abs(x1 - x2)) + (abs(y1 - y2)))
            if x == 6:
                state = node.state
                if state.index(6) == 0:
                    x1 = 1
                    y1 = 1
                if state.index(6) == 1:
                    x1 = 2
                    y1 = 1
                if state.index(6) == 2:
                    x1 = 3
                    y1 = 1
                if state.index(6) == 3:
                    x1 = 1
                    y1 = 2
                if state.index(6) == 4:
                    x1 = 2
                    y1 = 2
                if state.index(6) == 5:
                    x1 = 3
                    y1 = 2
                if state.index(6) == 6:
                    x1 = 1
                    y1 = 3
                if state.index(6) == 7:
                    x1 = 2
                    y1 = 3
               
                x2 = 3
                y2 = 2
                sum = sum + ((abs(x1 - x2)) + (abs(y1 - y2)))
            if x == 7:
                state = node.state
                if state.index(7) == 0:
                    x1 = 1
                    y1 = 1
                if state.index(7) == 1:
                    x1 = 2
                    y1 = 1
                if state.index(7) == 2:
                    x1 = 3
                    y1 = 1
                if state.index(7) == 3:
                    x1 = 1
                    y1 = 2
                if state.index(7) == 4:
                    x1 = 2
                    y1 = 2
                if state.index(7) == 5:
                    x1 = 3
                    y1 = 2
                if state.index(7) == 6:
                    x1 = 1
                    y1 = 3
                if state.index(7) == 7:
                    x1 = 2
                    y1 = 3
               
                x2 = 1
                y2 = 3
                sum = sum + ((abs(x1 - x2)) + (abs(y1 - y2)))
            if x == 8:
                state = node.state
                if state.index(8) == 0:
                    x1 = 1
                    y1 = 1
                if state.index(8) == 1:
                    x1 = 2
                    y1 = 1
                if state.index(8) == 2:
                    x1 = 3
                    y1 = 1
                if state.index(8) == 3:
                    x1 = 1
                    y1 = 2
                if state.index(8) == 4:
                    x1 = 2
                    y1 = 2
                if state.index(8) == 5:
                    x1 = 3
                    y1 = 2
                if state.index(8) == 6:
                    x1 = 1
                    y1 = 3
                if state.index(8) == 7:
                    x1 = 2
                    y1 = 3
               
                x2 = 2
                y2 = 3
                sum = sum + ((abs(x1 - x2)) + (abs(y1 - y2)))
            
        #adds formula sum each time and returns
        return sum





# ______________________________________________________________________________
#main

#frontier = a set of possible states that could exist given an action
#search method tracks node in the frontier
#node class has a pointer to each parent, so no tree object
#(1,5,2,3,4,6,8,7,0) shouldnt be more than a 16 move solution 

#10 puzzle problems
puzzle = EightPuzzle((1,5,2,3,4,6,8,7,0),(1, 2, 3, 4, 5, 6, 7, 8, 0))
puzzle2 = EightPuzzle((1,5,2,3,4,6,8,0,7),(1, 2, 3, 4, 5, 6, 7, 8, 0))
puzzle3 = EightPuzzle((0,2,4,1,6,3,8,7,5),(1, 2, 3, 4, 5, 6, 7, 8, 0))
puzzle4 = EightPuzzle((3,8,1,2,4,6,7,0,5),(1, 2, 3, 4, 5, 6, 7, 8, 0))
puzzle5 = EightPuzzle((1,6,5,8,4,0,3,2,7),(1, 2, 3, 4, 5, 6, 7, 8, 0))
puzzle6 = EightPuzzle((5,0,7,6,4,2,1,8,3),(1, 2, 3, 4, 5, 6, 7, 8, 0))
puzzle7 = EightPuzzle((2,7,6,0,5,1,4,3,8),(1, 2, 3, 4, 5, 6, 7, 8, 0))
puzzle8 = EightPuzzle((4,0,2,3,5,7,8,1,6),(1, 2, 3, 4, 5, 6, 7, 8, 0))
puzzle9 = EightPuzzle((2,4,0,6,8,1,7,3,5),(1, 2, 3, 4, 5, 6, 7, 8, 0))
puzzle10 = EightPuzzle((6,1,8,7,5,3,2,0,4),(1, 2, 3, 4, 5, 6, 7, 8, 0))

print("By Jacob Hitzges\n\n")

#solvability test
print("Testing check solvability method")
print(puzzle3.check_solvability((6,1,8,7,5,3,2,0,4)))
print("\n")

#action test
print("Testing Action Method")
print(puzzle.actions((1,5,2,3,4,6,8,7,0)))
print("\n")

#result test
print("Testing Result Method")
print(puzzle.result((1,5,2,3,4,6,8,7,0), 'LEFT'))
print(puzzle2.result((1,5,2,3,4,6,0,8,7), 'RIGHT'))
print(puzzle3.result((1,5,2,3,0,4,8,6,7), 'UP'))
print(puzzle3.result((1,5,2,3,0,4,8,6,7), 'DOWN'))
print("\n")

puzNode = Node((1,5,2,3,4,6,8,7,0))

#h and h2 test
print("Testing for h")  
print(puzzle.h(puzNode))
print("Testing for h2")
print(puzzle.h2(puzNode))
print("\n")

#calls A* for all puzzles
goal1, expandCount = astar_search(puzzle, h=puzzle.h)
goal12, expandCount2 = astar_search(puzzle, h=puzzle.h2)
goal2, expandCount = astar_search(puzzle2, h=puzzle.h)
goal22, expandCount2 = astar_search(puzzle2, h=puzzle.h2)
goal3, expandCount = astar_search(puzzle3, h=puzzle.h)
goal32, expandCount2 = astar_search(puzzle3, h=puzzle.h2)
goal4, expandCount = astar_search(puzzle4, h=puzzle.h)
goal42, expandCount2 = astar_search(puzzle4, h=puzzle.h2)
goal5, expandCount = astar_search(puzzle5, h=puzzle.h)
goal52, expandCount2 = astar_search(puzzle5, h=puzzle.h2)
goal6, expandCount = astar_search(puzzle6, h=puzzle.h)
goal62, expandCount2 = astar_search(puzzle6, h=puzzle.h2)
goal7, expandCount = astar_search(puzzle7, h=puzzle.h)
goal72, expandCount2 = astar_search(puzzle7, h=puzzle.h2)
goal8, expandCount = astar_search(puzzle8, h=puzzle.h)
goal82, expandCount2 = astar_search(puzzle8, h=puzzle.h2)
goal9, expandCount = astar_search(puzzle9, h=puzzle.h)
goal92, expandCount2 = astar_search(puzzle9, h=puzzle.h2)
goal10, expandCount = astar_search(puzzle10, h=puzzle.h)
goal102, expandCount2 = astar_search(puzzle10, h=puzzle.h2)

#prints starting state, steps and solution for all puzzles with A* search
print("Start State: ", puzzle.initial,"\n","h Solution Move Sequence\n",goal1.solution(),"\nSoluti.on State: ", goal1,"\n")
print("Start State: ", puzzle.initial,"\n","h2 Solution Move Sequence\n",goal12.solution(),"\nSolution State: ", goal12,"\n\n\n\n")
print("Start State: ", puzzle2.initial,"\n","h Solution Move Sequence\n",goal2.solution(),"\nSolution State: ", goal2,"\n")
print("Start State: ", puzzle2.initial,"\n","h2 Solution Move Sequence\n",goal22.solution(),"\nSolution State: ", goal22,"\n\n\n\n")
print("Start State: ", puzzle3.initial,"\n","h Solution Move Sequence\n",goal3.solution(),"\nSolution State: ", goal3,"\n")
print("Start State: ", puzzle3.initial,"\n","h2 Solution Move Sequence\n",goal32.solution(),"\nSolution State: ", goal32,"\n\n\n\n")
print("Start State: ", puzzle4.initial,"\n","h Solution Move Sequence\n",goal4.solution(),"\nSolution State: ", goal4,"\n")
print("Start State: ", puzzle4.initial,"\n","h2 Solution Move Sequence\n",goal42.solution(),"\nSolution State: ", goal42,"\n\n\n\n")
print("Start State: ", puzzle5.initial,"\n","h Solution Move Sequence\n",goal5.solution(),"\nSolution State: ", goal5,"\n")
print("Start State: ", puzzle5.initial,"\n","h2 Solution Move Sequence\n",goal52.solution(),"\nSolution State: ", goal52,"\n\n\n\n")
print("Start State: ", puzzle6.initial,"\n","h Solution Move Sequence\n",goal6.solution(),"\nSolution State: ", goal6,"\n")
print("Start State: ", puzzle6.initial,"\n","h2 Solution Move Sequence\n",goal62.solution(),"\nSolution State: ", goal62,"\n\n\n\n")
print("Start State: ", puzzle7.initial,"\n","h Solution Move Sequence\n",goal7.solution(),"\nSolution State: ", goal7,"\n")
print("Start State: ", puzzle7.initial,"\n","h2 Solution Move Sequence\n",goal72.solution(),"\nSolution State: ", goal72,"\n\n\n\n")
print("Start State: ", puzzle8.initial,"\n","h Solution Move Sequence\n",goal8.solution(),"\nSolution State: ", goal8,"\n")
print("Start State: ", puzzle8.initial,"\n","h2 Solution Move Sequence\n",goal82.solution(),"\nSolution State: ", goal82,"\n\n\n\n")
print("Start State: ", puzzle9.initial,"\n","h Solution Move Sequence\n",goal9.solution(),"\nSolution State: ", goal9,"\n")
print("Start State: ", puzzle9.initial,"\n","h2 Solution Move Sequence\n",goal92.solution(),"\nSolution State: ", goal92,"\n\n\n\n")
print("Start State: ", puzzle10.initial,"\n","h Solution Move Sequence\n",goal10.solution(),"\nSolution State: ", goal10,"\n")
print("Start State: ", puzzle10.initial,"\n","h2 Solution Move Sequence\n",goal102.solution(),"\nSolution State: ", goal102,"\n\n\n\n")

#H and H2 Time Testing
start = time.time()
goal1, expandCount = astar_search(puzzle, h=puzzle.h)
end = time.time()
print((end - start),"\n\n\n\n")

start = time.time()
goal12, expandCount2 = astar_search(puzzle, h=puzzle.h2)
end = time.time()
print((end - start),"\n\n\n\n")

start = time.time()
goal2, expandCount = astar_search(puzzle2, h=puzzle.h)
end = time.time()
print((end - start),"\n\n\n\n")

start = time.time()
goal22, expandCount2 = astar_search(puzzle2, h=puzzle.h2)
end = time.time()
print((end - start),"\n\n\n\n")

start = time.time()
goal3, expandCount = astar_search(puzzle3, h=puzzle.h)
end = time.time()
print((end - start),"\n\n\n\n")

start = time.time()
goal32, expandCount2 = astar_search(puzzle3, h=puzzle.h2)
end = time.time()
print((end - start),"\n\n\n\n")

start = time.time()
goal4, expandCount = astar_search(puzzle4, h=puzzle.h)
end = time.time()
print((end - start),"\n\n\n\n")

start = time.time()
goal42, expandCount2 = astar_search(puzzle4, h=puzzle.h2)
end = time.time()
print((end - start),"\n\n\n\n")

start = time.time()
goal5, expandCount = astar_search(puzzle5, h=puzzle.h)
end = time.time()
print((end - start),"\n\n\n\n")

start = time.time()
goal52, expandCount2 = astar_search(puzzle5, h=puzzle.h2)
end = time.time()
print((end - start),"\n\n\n\n")

start = time.time()
goal6, expandCount = astar_search(puzzle6, h=puzzle.h)
end = time.time()
print((end - start),"\n\n\n\n")

start = time.time()
goal62, expandCount = astar_search(puzzle6, h=puzzle.h)
end = time.time()
print((end - start),"\n\n\n\n")

start = time.time()
goal7, expandCount = astar_search(puzzle7, h=puzzle.h)
end = time.time()
print((end - start),"\n\n\n\n")

start = time.time()
goal72, expandCount = astar_search(puzzle7, h=puzzle.h)
end = time.time()
print((end - start),"\n\n\n\n")

start = time.time()
goal8, expandCount = astar_search(puzzle8, h=puzzle.h)
end = time.time()
print((end - start),"\n\n\n\n")

start = time.time()
goal82, expandCount = astar_search(puzzle8, h=puzzle.h)
end = time.time()
print((end - start),"\n\n\n\n")

start = time.time()
goal9, expandCount = astar_search(puzzle9, h=puzzle.h)
end = time.time()
print((end - start),"\n\n\n\n")

start = time.time()
goal92, expandCount = astar_search(puzzle9, h=puzzle.h)
end = time.time()
print((end - start),"\n\n\n\n")

start = time.time()
goal10, expandCount = astar_search(puzzle10, h=puzzle.h)
end = time.time()
print((end - start),"\n\n\n\n")

start = time.time()
goal102, expandCount = astar_search(puzzle10, h=puzzle.h)
end = time.time()
print((end - start),"\n\n\n\n")

#IDS Time Testing
start = time.time()
iterative_deepening_search(puzzle)
end = time.time()
print((end - start),"\n\n\n\n")

start = time.time()
iterative_deepening_search(puzzle2)
end = time.time()
print((end - start),"\n\n\n\n")

start = time.time()
iterative_deepening_search(puzzle4)
end = time.time()
print((end - start),"\n\n\n\n")

start = time.time()
iterative_deepening_search(puzzle9)
end = time.time()
print((end - start),"\n\n\n\n")






