import numpy as np
import matplotlib.pyplot as plt
import time
from IPython import display
import random
import math

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

    @staticmethod
    def iter(shape):
        for r in range(shape[0]):
            for c in range(shape[1]):
                yield Pos(r, c)

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

    def is_dead(self):
        return self.player_pos == self.minotaur_pos

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
    MINOTAUR_REWARD = -1000


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
        for player_pos in Pos.iter(self.maze.shape):
            if self.maze[player_pos.unpack()] == 1:
                continue

            for minotaur_pos in Pos.iter(self.maze.shape):
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

    def __moves(self, state, action):
        """ Makes a step in the maze, given a current position and an action.
            If the action STAY or an inadmissible action is used, the agent stays in place.

            :return next state index and corresponding transition prob.
        """

        if state.is_dead() or self.maze[state.player_pos.unpack()] == 2:
            return [state] # game over

        # Compute the future position given current (state, action)
        new_player_pos = state.player_pos + self.actions[action]
        # Is the future position an impossible one ?
        agent_hitting_maze_walls = not new_player_pos.within(self.maze.shape) or self.maze[new_player_pos.unpack()] == 1

        # If we can't move, we stay
        if agent_hitting_maze_walls:
            new_player_pos = state.player_pos

        minotaur_actions = [Pos(0, -1), Pos(0, 1), Pos(-1, 0), Pos(1, 0)]

        if self.minotaur_stay:
            minotaur_actions.append(Pos(0, 0))

        next_states = []
        # keep on picking a new move as long as the chosen move is impossible
        for minotaur_action in minotaur_actions:
            new_mino_pos = state.minotaur_pos + minotaur_action
            if new_mino_pos.within(self.maze.shape):
                next_s = State(new_player_pos, new_mino_pos)
                next_states.append(next_s)

        return next_states

    def __move(self, state, action):
        return random.choice(self.__moves(state, action))

    def __transitions(self):
        """ Computes the transition probabilities for every state action pair.
            :return numpy.tensor transition probabilities: tensor of transition
            probabilities of dimension S*S*A
        """
        # Initialize the transition probailities tensor (S,S,A)
        dimensions = (self.n_states,self.n_states,self.n_actions);
        transition_probabilities = np.zeros(dimensions);

        # Compute the transition probabilities
        for s in range(self.n_states):
            for a in range(self.n_actions):
                next_states = self.__moves(self.states[s], a)

                prob = 1.0 / len(next_states)

                for next_s in next_states:
                    transition_probabilities[self.map[next_s], s, a] = prob

        return transition_probabilities;

    def __rewards(self, weights=None, random_rewards=None):
        rewards = np.zeros((self.n_states, self.n_actions))

        # If the rewards are not described by a weight matrix
        if weights is None:
            for s in range(self.n_states):
                state = self.states[s]

                for a in range(self.n_actions):
                    next_states = self.__moves(state, a)

                    # calculate average expected reward over all possible outcomes
                    cumulative_reward = 0.0
                    for next_state in next_states:
                        agent_stayed = state.player_pos == next_state.player_pos

                        # Reward for hitting a wall
                        if self.maze[next_state.player_pos.unpack()] == 2:
                            cumulative_reward += self.GOAL_REWARD
                        # Reward for reaching the exit
                        # TODO: why does the agent need to stay to receive the goal reward?
                        elif next_state.player_pos == next_state.minotaur_pos:
                            cumulative_reward += self.MINOTAUR_REWARD
                        #TODO I'm not entirely sure if this is current
                        elif agent_stayed and a != self.STAY:
                            cumulative_reward += self.IMPOSSIBLE_REWARD
                        # Reward for taking a step to an empty cell that is not the exit
                        else:
                            cumulative_reward += self.STEP_REWARD

                    rewards[s,a] = cumulative_reward / len(next_states)

                        ## If there exists trapped cells with probability 0.5
                        #if random_rewards and self.maze[next_state.player_pos.unpack()]<0:
                        #    row, col = next_state.player_pos.unpack()
                        #    # With probability 0.5 the reward is
                        #    r1 = (1 + abs(self.maze[row, col])) * rewards[s,a]
                        #    # With probability 0.5 the reward is
                        #    r2 = rewards[s,a]
                        #    # The average reward
                        #    rewards[s,a] = 0.5*r1 + 0.5*r2
        # If the weights are described by a weight matrix
        else:
            for s in range(self.n_states):
                 for a in range(self.n_actions):
                    next_states = self.__moves(self.states[s], a)

                    # calculate average expected reward over all possible outcomes
                    cumulative_reward = 0.0
                    for next_state in next_states:
                        i,j = next_state.player_pos.unpack()
                        cumulative_reward += weights[i][j]

                    rewards[s,a] = cumulative_reward / len(next_states)

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
            s = start;
            # Add the starting position in the maze to the path
            path.append(start);
            while t < horizon-1:
                # Move to next state given the policy and the current state
                next_s = self.__move(s, policy[self.map[s],t]);
                # Add the position in the maze corresponding to the next state
                # to the path
                path.append(next_s)
                # Update time and state for next iteration
                t += 1
                s = next_s
        if method == 'ValIter':
            # Initialize current state, next state and time
            t = 1;
            s = start;
            # Add the starting position in the maze to the path
            path.append(start);
            # Move to next state given the policy and the current state
            next_s = self.__move(s, policy[self.map[s]]);
            # Add the position in the maze corresponding to the next state
            # to the path
            path.append(next_s);
            # Loop while state is not the goal state
            while s != next_s:
                # Update state
                s = next_s;
                # Move to next state given the policy and the current state
                next_s = self.__move(s, policy[s]);
                # Add the position in the maze corresponding to the next state
                # to the path
                path.append(next_s)
                # Update time and state for next iteration
                t +=1;
        return path

    def survival_rate(self, start, policy, method, exit, num = 10000):
        won = 0;
        for i in range(num):
            path = self.simulate(start, policy, method);
            if path[-1].player_pos == exit:
                won += 1;
        
        return won/num

    def survival_rate2(self, start, policy, method, exit):
        if method not in methods:
            error = 'ERROR: the argument method must be in {}'.format(methods);
            raise NameError(error);

        states = set()
        num_states = dict()
        prob_states = dict()

        states.add(start)
        num_states[start] = 1
        prob_states[start] = 1

        total_num_states = 1

        horizon = policy.shape[1];

        t = 0
        while t < horizon-1:
            next_states = set()
            next_num = dict()
            next_prob = dict()
            next_total = 0

            #print("States at {}: unique {}, total {}".format(t, len(states), total_num_states))

            for s in states:
                s_count = num_states[s]
                s_prob = prob_states[s]

                action = policy[self.map[s], t]
                new_states = self.__moves(s, action)

                each_ns_prob = s_prob / len(new_states)

                for ns in new_states:
                    next_num[ns] = next_num.get(ns, 0) + s_count
                    next_prob[ns] = next_prob.get(ns, 0) + each_ns_prob

                    next_states.add(ns)
                    next_total += s_count

            # nothing moving, break
            if next_states == states:
                break

            states = next_states
            num_states = next_num
            prob_states = next_prob
            total_num_states = next_total

            t += 1


        won = 0
        dead = 0
        for state, prob in prob_states.items():
            if state.player_pos == exit:
                won += prob
            elif state.is_dead():
                dead += prob

        print("T = {}, win {:%}, dead {:%}".format(horizon-1, won, dead))

        return won

    def show(self):
        print('The states are :')
        print(self.states)
        print('The actions are:')
        print(self.actions)
        print('The mapping of the states:')
        print(self.map)
        print('The rewards:')
        print(self.rewards)

    def animate_solution(self, path, policy = None):
        # the minotaur starts at the exit

        # Map a color to each cell in the maze
        col_map = {0: WHITE, 1: BLACK, 2: LIGHT_GREEN, -6: LIGHT_RED, -1: LIGHT_RED}

        # Size of the maze
        rows, cols = self.maze.shape

        # Create figure of the size of the maze
        fig = plt.figure(1, figsize=(cols,rows))

        # Remove the axis ticks and add title title
        ax = plt.gca()
        ax.set_title('Policy simulation')
        ax.set_xticks([])
        ax.set_yticks([])

        # Give a color to each cell
        colored_maze = [[col_map[self.maze[j,i]] for i in range(cols)] for j in range(rows)];

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

        def draw_action_in_cell(pos, action):
            if action == self.STAY:
                return None

            cell = grid.get_celld()[pos]

            arrow_size_x = cell.get_width()*0.33
            arrow_size_y = cell.get_width()*0.33

            cell_mid_x = cell.get_x() + 0.5*cell.get_width()
            cell_mid_y = cell.get_y() + 0.5*cell.get_height()

            dirs = dict()
            dirs[self.MOVE_DOWN] = (0, -arrow_size_y)
            dirs[self.MOVE_UP] = (0, arrow_size_y)
            dirs[self.MOVE_RIGHT] = (arrow_size_x, 0)
            dirs[self.MOVE_LEFT] = (-arrow_size_x, 0)

            dx, dy = dirs[action]

            #print(cell_mid_x, cell_mid_y, dx, dy)

            return plt.arrow(cell_mid_x - dx/2, cell_mid_y - dy/2, dx, dy, width = 0.005)

        arrows = []

        for t, s in enumerate(path):
            player_tuple = s.player_pos.unpack()
            minotaur_tuple = s.minotaur_pos.unpack()

            if prev_player_tuple:
                if not out:
                    # set previous player position back to white
                    grid.get_celld()[prev_player_tuple].set_facecolor(col_map[self.maze[prev_player_tuple]])
                    grid.get_celld()[prev_player_tuple].get_text().set_text('')
                if self.maze[player_tuple] == 2:
                    grid.get_celld()[player_tuple].set_facecolor(LIGHT_GREEN)
                    grid.get_celld()[player_tuple].get_text().set_text('Player is out')
                    out = True

                # set previous minotaur position back to white
                grid.get_celld()[prev_minotaur_tuple].set_facecolor(col_map[self.maze[prev_minotaur_tuple]])
                grid.get_celld()[prev_minotaur_tuple].get_text().set_text('')

            grid.get_celld()[player_tuple].set_facecolor(LIGHT_ORANGE)
            grid.get_celld()[player_tuple].get_text().set_text('Player')
            grid.get_celld()[minotaur_tuple].set_facecolor(LIGHT_PURPLE)
            grid.get_celld()[minotaur_tuple].get_text().set_text('Minotaur')

            prev_player_tuple, prev_minotaur_tuple = player_tuple, minotaur_tuple

            if policy is not None:
                for a in arrows:
                    if a is not None:
                        a.remove()
                arrows = []

                for draw_pos in Pos.iter(self.maze.shape):
                    if draw_pos == s.minotaur_pos or self.maze[draw_pos.unpack()] != 0:
                        continue

                    draw_state = State(draw_pos, s.minotaur_pos)
                    if policy.ndim == 1:
                        draw_action = policy[self.map[draw_state]]
                    else:
                        draw_action = policy[self.map[draw_state], t]

                    arrow = draw_action_in_cell(draw_pos.unpack(), draw_action)
                    if arrow:
                        arrows.append(arrow)


            # This animation only works in ipython notebook
            display.display(fig)
            #plt.clear_output(wait=True)
            display.clear_output(wait=True)
            time.sleep(1)
        #plt.show()

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

def survival_rates(maze, exit, T_range, minotaur_stay, num = 10000):
    env = Maze(maze, minotaur_stay = minotaur_stay)

    sr = []

    method = "DynProg"
    start = State(Pos(0,0), exit)

    for T in T_range:
        V, policy = dynamic_programming(env, T)

        sr.append(env.survival_rate2(start, policy, method, exit))

    return sr