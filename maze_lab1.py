import numpy as np
import matplotlib.pyplot as plt
import time
from IPython import display
import random

# Implemented methods
methods = ['DynProg', 'ValIter'];

# Some colours
LIGHT_RED    = '#FFC4CC';
LIGHT_GREEN  = '#95FD99';
BLACK        = '#000000';
WHITE        = '#FFFFFF';
LIGHT_PURPLE = '#E8D0FF';
LIGHT_ORANGE = '#FAE0C3';

class Pos:
    def __init__(self, r, c):
        self.row = r
        self.col = c

    def __add__(self, o):
        return Pos(self.row + o.row, self.col + o.col)

    def __hash__(self):
        return hash((self.row, self.col))

    def __eq__(self, o):
        return self.row == o.row and self.col == o.col

    def __str__(self):
        return "({},{})".format(self.row, self.col)

    def unpack(self):
        return self.row, self.col

    def within(self, shape):
        return self.row >= 0 and self.col >= 0 and self.row < shape[0] and self.col < shape[1]

class State:
    def __init__(self, player_pos, minotaur_pos):
        self.player_pos = player_pos
        self.minotaur_pos = minotaur_pos

    def __hash__(self):
        return hash((self.player_pos, self.minotaur_pos))

    def __eq__(self, o):
        return self.player_pos == o.player_pos and self.minotaur_pos == o.minotaur_pos

    def __str__(self):
        return "Player: {}, Minotaur: {}".format(self.player_pos, self.minotaur_pos)

class Maze:

    # Actions
    STAY       = 0
    MOVE_LEFT  = 1
    MOVE_RIGHT = 2
    MOVE_UP    = 3
    MOVE_DOWN  = 4

    # Give names to actions
    actions_names = {
        STAY: "stay",
        MOVE_LEFT: "move left",
        MOVE_RIGHT: "move right",
        MOVE_UP: "move up",
        MOVE_DOWN: "move down"
    }

    # Reward values
    STEP_REWARD = 0
    GOAL_REWARD = 1
    IMPOSSIBLE_REWARD = -100
    MINOTAUR_REWARD = -10


    def __init__(self, maze, weights=None, random_rewards=False, minotaur_stay=False):
        """ Constructor of the environment Maze.
        """
        self.maze                     = maze;
        self.minotaur_stay            = minotaur_stay
        self.actions                  = self.__actions();
        self.states, self.map         = self.__states();
        self.n_actions                = len(self.actions);
        self.n_states                 = len(self.states);
        self.transition_probabilities = self.__transitions();
        self.rewards                  = self.__rewards(weights=weights,
                                                random_rewards=random_rewards);


    def __actions(self):
        actions = dict();
        actions[self.STAY]       = Pos(0, 0);
        actions[self.MOVE_LEFT]  = Pos(0,-1);
        actions[self.MOVE_RIGHT] = Pos(0, 1);
        actions[self.MOVE_UP]    = Pos(-1,0);
        actions[self.MOVE_DOWN]  = Pos(1,0);
        return actions;

    def __states(self):
        states = dict();
        map = dict();
        s = 0;
        for i in range(self.maze.shape[0]):
            for j in range(self.maze.shape[1]):
                if self.maze[i,j] == 1:
                    continue

                player_pos = Pos(i,j)

                for k in range(self.maze.shape[0]):
                    for l in range(self.maze.shape[1]):
                        minotaur_pos = Pos(k,l)

                        new_state = State(player_pos, minotaur_pos)

                        states[s] = new_state
                        map[new_state] = s;

                        s += 1;
        # terminal states for "Eaten by the minotaur" and "Escaped"
        # states[s] = "EATEN"
        # states[s+1] = "ESCAPED"
        # map["EATEN"] = s
        # map["ESCAPED"] = s+1
        return states, map

    def __move(self, state, action):
        """ Makes a step in the maze, given a current position and an action.
            If the action STAY or an inadmissible action is used, the agent stays in place.

            :return next state index and corresponding transition prob.
        """

        state_obj = self.states[state]

        # Compute the future position given current (state, action)
        new_player_pos = state_obj.player_pos + self.actions[action]
        # Is the future position an impossible one ?
        agent_hitting_maze_walls = not new_player_pos.within(self.maze.shape) or self.maze[new_player_pos.unpack()] == 1

        minotaur_actions = [Pos(0, -1), Pos(0, 1), Pos(-1, 0), Pos(1, 0)]

        if self.minotaur_stay:
            minotaur_actions.append(Pos(0, 0))

        valid_minotaur_pos = []
        # keep on picking a new move as long as the chosen move is impossible
        for minotaur_action in minotaur_actions:
            new_mino_pos = state_obj.minotaur_pos + minotaur_action
            if new_mino_pos.within(self.maze.shape):
                valid_minotaur_pos.append(new_mino_pos)

        actual_new_mino_pos = random.choice(valid_minotaur_pos)

        move_prob = 1.0 / len(valid_minotaur_pos)

        # Based on the impossibility check return the next state.
        if agent_hitting_maze_walls:
            # the agent does not move but the minotaur does
            return self.map[State(state_obj.player_pos, actual_new_mino_pos)], move_prob
        else:
            return self.map[State(new_player_pos, actual_new_mino_pos)], move_prob

    def __transitions(self):
        """ Computes the transition probabilities for every state action pair.
            :return numpy.tensor transition probabilities: tensor of transition
            probabilities of dimension S*S*A
        """
        # Initialize the transition probailities tensor (S,S,A)
        dimensions = (self.n_states,self.n_states,self.n_actions);
        transition_probabilities = np.zeros(dimensions);

        # Compute the transition probabilities. Note that the transitions
        # are deterministic.
        for s in range(self.n_states):
            for a in range(self.n_actions):
                next_s, prob = self.__move(s,a)
                transition_probabilities[next_s, s, a] = prob
        return transition_probabilities;

    def __rewards(self, weights=None, random_rewards=None):
        rewards = np.zeros((self.n_states, self.n_actions))

        # If the rewards are not described by a weight matrix
        if weights is None:
            for s in range(self.n_states):
                state = self.states[s]

                for a in range(self.n_actions):
                    next_s, prob = self.__move(s,a)
                    next_state = self.states[next_s]

                    agent_stayed = state.player_pos == next_state.player_pos
                    # Reward for hitting a wall
                    if agent_stayed and a != self.STAY:
                        rewards[s,a] = self.IMPOSSIBLE_REWARD
                    # Reward for reaching the exit
                    # TODO: why does the agent need to stay to receive the goal reward?
                    elif agent_stayed and self.maze[next_state.player_pos.unpack()] == 2:
                        rewards[s,a] = self.GOAL_REWARD
                    #TODO I'm not entirely sure if this is current
                    elif next_state.player_pos == next_state.minotaur_pos:
                        rewards[s,a] = self.MINOTAUR_REWARD
                    # Reward for taking a step to an empty cell that is not the exit
                    else:
                        rewards[s,a] = self.STEP_REWARD

                    # If there exists trapped cells with probability 0.5
                    if random_rewards and self.maze[next_state.player_pos.unpack()]<0:
                        row, col = next_state.player_pos.unpack()
                        # With probability 0.5 the reward is
                        r1 = (1 + abs(self.maze[row, col])) * rewards[s,a]
                        # With probability 0.5 the reward is
                        r2 = rewards[s,a]
                        # The average reward
                        rewards[s,a] = 0.5*r1 + 0.5*r2
        # If the weights are described by a weight matrix
        else:
            for s in range(self.n_states):
                 for a in range(self.n_actions):
                    next_s, prob = self.__move(s,a);
                    next_state = self.states[next_s]
                    i,j = next_state.player_pos.unpack()
                    # Simply put the reward as the weights o the next state.
                    rewards[s,a] = weights[i][j]
        return rewards

    def simulate(self, start, policy, method):
        if method not in methods:
            error = 'ERROR: the argument method must be in {}'.format(methods);
            raise NameError(error);

        path = list();
        if method == 'DynProg':
            # Deduce the horizon from the policy shape
            horizon = policy.shape[1];
            # Initialize current state and time
            t = 0;
            s = self.map[start];
            # Add the starting position in the maze to the path
            path.append(start);
            while t < horizon-1:
                # Move to next state given the policy and the current state
                next_s, prob = self.__move(s, policy[s,t]);
                # Add the position in the maze corresponding to the next state
                # to the path
                path.append(self.states[next_s])
                # Update time and state for next iteration
                t += 1
                s = next_s
        if method == 'ValIter':
            # Initialize current state, next state and time
            t = 1;
            s = self.map[start];
            # Add the starting position in the maze to the path
            path.append(start);
            # Move to next state given the policy and the current state
            next_s = self.__move(s,policy[s]);
            # Add the position in the maze corresponding to the next state
            # to the path
            path.append(self.states[next_s]);
            # Loop while state is not the goal state
            while s != next_s:
                # Update state
                s = next_s;
                # Move to next state given the policy and the current state
                next_s = self.__move(s,policy[s]);
                # Add the position in the maze corresponding to the next state
                # to the path
                path.append(self.states[next_s])
                # Update time and state for next iteration
                t +=1;
        return path


    def show(self):
        print('The states are :')
        print(self.states)
        print('The actions are:')
        print(self.actions)
        print('The mapping of the states:')
        print(self.map)
        print('The rewards:')
        print(self.rewards)

def dynamic_programming(env, horizon):
    """ Solves the shortest path problem using dynamic programming
        :input Maze env           : The maze environment in which we seek to
                                    find the shortest path.
        :input int horizon        : The time T up to which we solve the problem.
        :return numpy.array V     : Optimal values for every state at every
                                    time, dimension S*T
        :return numpy.array policy: Optimal time-varying policy at every state,
                                    dimension S*T
    """

    # The dynamic prgramming requires the knowledge of :
    # - Transition probabilities
    # - Rewards
    # - State space
    # - Action space
    # - The finite horizon
    p         = env.transition_probabilities;
    r         = env.rewards;
    n_states  = env.n_states;
    n_actions = env.n_actions;
    T         = horizon;

    # The variables involved in the dynamic programming backwards recursions
    V      = np.zeros((n_states, T+1));
    policy = np.zeros((n_states, T+1));
    Q      = np.zeros((n_states, n_actions));


    # Initialization
    Q            = np.copy(r);
    V[:, T]      = np.max(Q,1);
    policy[:, T] = np.argmax(Q,1);

    # The dynamic programming backwards recursion
    for t in range(T-1,-1,-1):
        # Update the value function according to the bellman equation
        for s in range(n_states):
            for a in range(n_actions):
                # Update of the temporary Q values
                Q[s,a] = r[s,a] + np.dot(p[:,s,a],V[:,t+1])
        # Update by taking the maximum Q value w.r.t the action a
        V[:,t] = np.max(Q,1);
        # The optimal action is the one that maximizes the Q function
        policy[:,t] = np.argmax(Q,1);
    return V, policy;

def value_iteration(env, gamma, epsilon):
    """ Solves the shortest path problem using value iteration
        :input Maze env           : The maze environment in which we seek to
                                    find the shortest path.
        :input float gamma        : The discount factor.
        :input float epsilon      : accuracy of the value iteration procedure.
        :return numpy.array V     : Optimal values for every state at every
                                    time, dimension S*T
        :return numpy.array policy: Optimal time-varying policy at every state,
                                    dimension S*T
    """
    # The value itearation algorithm requires the knowledge of :
    # - Transition probabilities
    # - Rewards
    # - State space
    # - Action space
    # - The finite horizon
    p         = env.transition_probabilities;
    r         = env.rewards;
    n_states  = env.n_states;
    n_actions = env.n_actions;

    # Required variables and temporary ones for the VI to run
    V   = np.zeros(n_states);
    Q   = np.zeros((n_states, n_actions));
    BV  = np.zeros(n_states);
    # Iteration counter
    n   = 0;
    # Tolerance error
    tol = (1 - gamma)* epsilon/gamma;

    # Initialization of the VI
    for s in range(n_states):
        for a in range(n_actions):
            Q[s, a] = r[s, a] + gamma*np.dot(p[:,s,a],V);
    BV = np.max(Q, 1);

    # Iterate until convergence
    while np.linalg.norm(V - BV) >= tol and n < 200:
        # Increment by one the numbers of iteration
        n += 1;
        # Update the value function
        V = np.copy(BV);
        # Compute the new BV
        for s in range(n_states):
            for a in range(n_actions):
                Q[s, a] = r[s, a] + gamma*np.dot(p[:,s,a],V);
        BV = np.max(Q, 1);
        # Show error
        #print(np.linalg.norm(V - BV))

    # Compute policy
    policy = np.argmax(Q,1);
    # Return the obtained policy
    return V, policy;

def draw_maze(maze):

    # Map a color to each cell in the maze
    col_map = {0: WHITE, 1: BLACK, 2: LIGHT_GREEN, -6: LIGHT_RED, -1: LIGHT_RED};

    # Give a color to each cell
    rows,cols    = maze.shape;
    colored_maze = [[col_map[maze[j,i]] for i in range(cols)] for j in range(rows)];

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols,rows));

    # Remove the axis ticks and add title title
    ax = plt.gca();
    ax.set_title('The Maze');
    ax.set_xticks([]);
    ax.set_yticks([]);

    # Give a color to each cell
    rows,cols    = maze.shape;
    colored_maze = [[col_map[maze[j,i]] for i in range(cols)] for j in range(rows)];

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols,rows))

    # Create a table to color
    grid = plt.table(cellText=None,
                            cellColours=colored_maze,
                            cellLoc='center',
                            loc=(0,0),
                            edges='closed');
    # Modify the hight and width of the cells in the table
    tc = grid.properties()['child_artists']
    for cell in tc:
        cell.set_height(1.0/rows);
        cell.set_width(1.0/cols);

def animate_solution(maze, path):
    # the minotaur starts at the exit

    print(map(str, path))

    exit = path[0].minotaur_pos
    print("EXIT = ", exit)

    # Map a color to each cell in the maze
    col_map = {0: WHITE, 1: BLACK, 2: LIGHT_GREEN, -6: LIGHT_RED, -1: LIGHT_RED}

    # Size of the maze
    rows, cols = maze.shape

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols,rows))

    # Remove the axis ticks and add title title
    ax = plt.gca()
    ax.set_title('Policy simulation')
    ax.set_xticks([])
    ax.set_yticks([])

    # Give a color to each cell
    colored_maze = [[col_map[maze[j,i]] for i in range(cols)] for j in range(rows)];

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols, rows))

    # Create a table to color
    grid = plt.table(cellText=None,
                     cellColours=colored_maze,
                     cellLoc='center',
                     loc=(0,0),
                     edges='closed');

    # Modify the height and width of the cells in the table
    tc = grid.properties()['child_artists']
    for cell in tc:
        cell.set_height(1.0/rows);
        cell.set_width(1.0/cols);


    # Update the color at each frame
    out = False

    prev_player_tuple, prev_minotaur_tuple = None, None

    for s in path:
        player_tuple = s.player_pos.unpack()
        minotaur_tuple = s.minotaur_pos.unpack()

        if prev_player_tuple:
            if not out:
                # set previous player position back to white
                grid.get_celld()[prev_player_tuple].set_facecolor(col_map[maze[prev_player_tuple]])
                grid.get_celld()[prev_player_tuple].get_text().set_text('')
            if player_tuple == exit.unpack():
                grid.get_celld()[player_tuple].set_facecolor(LIGHT_GREEN)
                grid.get_celld()[player_tuple].get_text().set_text('Player is out')
                out = True

            # set previous minotaur position back to white
            grid.get_celld()[prev_minotaur_tuple].set_facecolor(col_map[maze[prev_minotaur_tuple]])
            grid.get_celld()[prev_minotaur_tuple].get_text().set_text('')

        grid.get_celld()[player_tuple].set_facecolor(LIGHT_ORANGE)
        grid.get_celld()[player_tuple].get_text().set_text('Player')
        grid.get_celld()[minotaur_tuple].set_facecolor(LIGHT_PURPLE)
        grid.get_celld()[minotaur_tuple].get_text().set_text('Minotaur')

        prev_player_tuple, prev_minotaur_tuple = player_tuple, minotaur_tuple 

        # This animation only works in ipython notebook
        display.display(fig)
        #plt.clear_output(wait=True)
        display.clear_output(wait=True)
        time.sleep(1)
    #plt.show()
