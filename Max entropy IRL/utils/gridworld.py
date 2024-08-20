import numpy as np
from itertools import product

class GridWorld:
    def __init__(self, size):
        self.size = size
        self.actions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        self.n_states = size**2
        self.n_actions = len(self.actions)
        self.p_transition = self._transition_prob_table()

    def state_index_to_point(self, state):
        return (state % self.size, state // self.size)
    
    def state_point_to_index(self, point):
        return point[0] + point[1] * self.size
    
    def state_point_to_index_clipped(self, state):
        s = (max(0, min(state[0], self.size - 1)), max(0, min(state[1], self.size - 1)))
        return self.state_point_to_index(s)
    
    def state_index_transition(self, s, a):
        s = self.state_index_to_point(s)
        s = (s[0] + self.actions[a][0], s[1] + self.actions[a][1])
        return self.state_point_to_index_clipped(s)

    def _transition_prob_table(self):
        table = np.zeros(shape=(self.n_states, self.n_states, self.n_actions))
        s1, s2, a = range(self.n_states), range(self.n_states), range(self.n_actions)

        for s_from, s_to, a in product(s1, s2, a):
            table[s_from, s_to, a] = self._transition_prob(s_from, s_to, a)
        
        return table
    
    def _transition_prob(self, s_from, s_to, a):
        fx, fy = self.state_index_to_point(s_from)
        tx, ty = self.state_index_to_point(s_to)
        ax, ay = self.actions[a]

        #deterministic
        if(fx + ax == tx and fy + ay == ty):
            return 1.0
        if(fx == tx and fy == ty):
            if not 0 <= fx + ax < self.size or not 0 <= fy + ay < self.size:
                return 1.0
            
        return 0.0
    
    def __repr__(self):
        return "GridWorld(size={})".format(self.size)
    

class IcyGridWorld(GridWorld):
    def __init__(self, size, p_slip=0.20):
        self.p_slip = p_slip
        super().__init__(size)

    def _transition_prob(self, s_from, s_to, a):
        fx, fy = self.state_index_to_point(s_from)
        tx, ty = self.state_index_to_point(s_to)
        ax, ay = self.actions[a]

        if( fx + ax == tx and fy + ay == ty):
            return 1.0 - self.p_slip + self.p_slip / self.n_actions
        
        if( abs(fx-tx) + abs(fy-ty) == 1):
            return self.p_slip / self.n_actions
        
         # we can stay at the same state if we would move over an edge
        if fx == tx and fy == ty:
            # intended move over an edge
            if not 0 <= fx + ax < self.size or not 0 <= fy + ay < self.size:
                # double slip chance at corners
                if not 0 < fx < self.size - 1 and not 0 < fy < self.size - 1:
                    return 1.0 - self.p_slip + 2.0 * self.p_slip / self.n_actions

                # regular probability at normal edges
                return 1.0 - self.p_slip + self.p_slip / self.n_actions

            # double slip chance at corners
            if not 0 < fx < self.size - 1 and not 0 < fy < self.size - 1:
                return 2.0 * self.p_slip / self.n_actions

            # single slip chance at edge
            if not 0 < fx < self.size - 1 or not 0 < fy < self.size - 1:
                return self.p_slip / self.n_actions

            # otherwise we cannot stay at the same state
            return 0.0

        # otherwise this transition is impossible
        return 0.0
    
    def __repr__(self):
        return "IcyGridWorld(size={}, p_slip={})".format(self.size, self.p_slip)
    

def state_features(world):
    return np.identity(world.n_states)

def coordinate_features(world):
    features = np.zeros((world.n_states, world.size))

    for s in range(world.n_states):
        x, y = world.state_index_to_point(s)
        features[s, x] += 1
        features[s, y] += 1

    return features