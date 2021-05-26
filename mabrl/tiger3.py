"""The new tiger implementation with 3 doors"""

# from general_bayes_adaptive_pomdps.domains import Domain
 

# class Tiger3(Domain):
#    """Tiger domain with 3 doors

#    Must implement interface of `Domain`
#    """
# this is the version adopted from used one, to see if my understanding has something wrong
    
from logging import Logger
from typing import List, Optional

import numpy as np

from general_bayes_adaptive_pomdps.core import (
    ActionSpace,
    DomainStepResult,
    SimulationResult,
)
from general_bayes_adaptive_pomdps.domains import domain
from general_bayes_adaptive_pomdps.misc import DiscreteSpace, LogLevel
# Discrete Space用处：
#
# 改造成三门
class Tiger3(domain.Domain):
    """The actual domain"""
    
    # consts
    # why no right?
    LEFT = 0
    # Listen number should be the one different from the state
    # state has the number 0, 1, 2
    # we can choose 4 here
    # since action has 4, we use 0, 1, 2 as open doors,then 3 as listening action
    LISTEN = 3
    
    # while there is a tiger the door opened, good reward gained, otherwise bad reward
    
    GOOD_DOOR_REWARD = 10
    BAD_DOOR_REWARD = -100
    # every time listen
    LISTEN_REWARD = -1
    # 改成三个，L, R, Mid
    ELEM_TO_STRING = ["L", "R", "Mid"]
    # As adviced, we can just remove the variable "one_hot_encode_observation"    
    # since it is not important to us

    def __init__(
        self,
        # one_hot_encode_observation: bool = 1,
        correct_obs_probs: Optional[List[float]] = None,
    ):
        """Construct the tiger domain
        Args:
             one_hot_encode_observation: (`bool`):
             correct_obs_probs: (`Optional[List[float]]`):
        """
        # 可以假设正确的概率全部为0.85
        # p(hear tiger left| tiger left) == 0.85
        # p(hear tiger middle| tiger middle) == 0.85
        # p(hear tiger right| tiger right) == 0.85
        # 默认设置
        if not correct_obs_probs:
            correct_obs_probs = [0.85, 0.85, 0.85]
            
        # 再加一项assert
        assert (
            0 <= correct_obs_probs[0] <= 1
        ), f"observation prob {correct_obs_probs[0]} not a probability"
        assert (
            0 <= correct_obs_probs[1] <= 1
        ), f"observation prob {correct_obs_probs[1]} not a probability"
        assert (
            0 <= correct_obs_probs[2] <= 1
        ), f"observation prob {correct_obs_probs[2]} not a probability"
        
        # Logger 用法
        # Logging 控制输出
        self._logger = Logger(self.__class__.__name__)
        
        self._correct_obs_probs = correct_obs_probs
        # since one_hot_encode_observation not used, this item can be no used any more
        
        # self._use_one_hot_obs = one_hot_encode_observation
        
        # states represent the number of doors so the amount fo states should be 3
        
        self._state_space = DiscreteSpace([3])

        # actions are listening and oprning these three doors
        # In this situation should have 4 actions: listen, open left door, open mid door and open right door
        
        self._action_space = ActionSpace(4)

        # 查明encode observation的作用
        # since we just focus on the our observation of tiger-3-door issue.
        
        self._obs_space = DiscreteSpace([4])

        # 采样单个状态
        self._state = self.sample_start_state()
        
    #可读
    #读取采样的初始状态
    @property
    def state(self):
        """ returns current state """
        return self._state
    
    # state.setter
    # 转换为setter:可写
    # define single state by yourself
    @state.setter
    def state(self, state: np.ndarray):
        """sets state
        Args:
             state: (`np.ndarray`): [0] or [1]
        """
        # to ensure something
        # state[0] should be just [0], [1] and [2]
        assert state.shape == (1,), f"{state} not correct shape"
        assert 3 > state[0] >= 0, f"{state} not valid"
        
        self._state = state
        
        
    @property
    def state_space(self) -> DiscreteSpace:
        """ a `general_bayes_adaptive_pomdps.misc.DiscreteSpace` ([2]) space """
        return self._state_space
    
    @property
    def action_space(self) -> ActionSpace:
        """ a `general_bayes_adaptive_pomdps.core.ActionSpace` ([3]) space """
        return self._action_space
    
    @property
    def observation_space(self) -> DiscreteSpace:
        """ a `general_bayes_adaptive_pomdps.misc.DiscreteSpace` ([1,1]) space if one-hot, otherwise [3]"""
        return self._obs_space
    
    # 编译观测
    def encode_observation(self, observation: int) -> np.ndarray:
        """encodes the observation for usage outside of `this`
        This wraps the `int` observation into a numpy array. Either directly,
        or with one-hot encoding if `Tiger` was initiated with that parameter
        to true
        Args:
            observation: (`int`): 0, 1= hear behind door, 2=null
        RETURNS (`np.ndarray`):
        """
        # here we don't use hot observation, then we just return observation
        # observation 0, 1, 2 = hear behind door
            
        return np.array([observation], dtype=int)
        
    # 静态方法
    # randomly sample one state from 0, 1, 2. This can be used to get next state.
    @staticmethod
    def sample_start_state() -> np.ndarray:
        """samples a random state (tiger left or right)
        RETURNS (`np.narray`): an initial state (in [[0],[1]])
        """
        #区间为[0,2)
        #取样的话应该是状态空间的值即[0，3)
        # which can be directly used to get the next state
        return np.array([np.random.randint(0, 3)], dtype=int)
    
    #采样observation
    def sample_observation(self, loc: int, listening: bool) -> int:
        """samples an observation, listening stores whether agent is listening
        Args:
             loc: (`int`): 0 is tiger left, 1 is tiger right
             listening: (`bool`): whether the agent is listening
        RETURNS (`int`): the observation: 0 = left, 1 = right, 2 = null
        """
        # here if 3 == null
        # so if not listening menas "listening:False"
        if not listening:
            return self.LISTEN
        
        
        return (
            loc if np.random.random() < self._correct_obs_probs[loc] else int(not loc)
        )
    
    def reset(self) -> np.ndarray:
        """Resets internal state and return first observation
        Resets the internal state randomly ([0] or [1])
        Returns [1,1] as a 'null' initial observation
        """
        self._state = self.sample_start_state()
        # when listen == 3, how will the encode_observation begins
        # encode_observation(3) means initial state? print to see if there is something happens
        return self.encode_observation(self.LISTEN)
    
    # 仿真步骤
    def simulation_step(self, state: np.ndarray, action: int) -> SimulationResult:
        """Simulates stepping from state using action. Returns interaction
        Will terminate episode when action is to open door,
        otherwise return an observation.
        Args:
             state: (`np.ndarray`): [0] is tiger left, [1] is tiger right
             action: (`int`): 0 is open left, 1 is open right or 2 is listen
        RETURNS (`general_bayes_adaptive_pomdps.core.SimulationResult`): the transition
        """
        # correction listen in tiger probelm is 0 or 1, LISTEN == 2
        # not equal to self.LISTEN means we have listen or open door.
        
        
        if action != self.LISTEN:
            obs = self.sample_observation(state[0], False)
            # return value is 3, which means null, no observation
            new_state = self.sample_start_state()
            
        else:
            # not opening door
            obs = self.sample_observation(state[0], True)
            new_state = state.copy()
        
        return SimulationResult(new_state, self.encode_observation(obs))
    
    # reward
    # 开门对错有奖励
    def reward(self, state: np.ndarray, action: int, new_state: np.ndarray) -> float:
        """A constant if listening, penalty if opening to door, and reward otherwise
        Args:
             state: (`np.ndarray`):
             action: (`int`):
             new_state: (`np.ndarray`):
        RETURNS (`float`):
        """
        
        assert self.state_space.contains(state), f"{state} not in space"
        assert self.state_space.contains(new_state), f"{new_state} not in space"
        assert self.action_space.contains(action), f"{action} not in space"
        
        if action == self.LISTEN:
            return self.LISTEN_REWARD
        
        # why state[0] instead of state?
        return self.GOOD_DOOR_REWARD if action == state[0] else self.BAD_DOOR_REWARD
    
    # 中止指令
    def terminal(self, state: np.ndarray, action: int, new_state: np.ndarray) -> bool:
        """True if opening a door
        Args:
             state: (`np.ndarray`):
             action: (`int`):
             new_state: (`np.ndarray`):
        RETURNS (`bool`):
        """
        # to ensure the state space contains new ones and old ones
        assert self.state_space.contains(state), f"{state} not in space"
        assert self.state_space.contains(new_state), f"{new_state} not in space"
        assert self.action_space.contains(action), f"{action} not in space"
        
        return bool(action != self.LISTEN)
    
    def step(self, action: int) -> DomainStepResult:
        """Performs a step in the tiger problem given action
        Will terminate episode when action is to open door,
        otherwise return an observation.
        Args:
             action: (`int`): 0 is open left, 1 is open right or 2 is listen
        RETURNS (`general_bayes_adaptive_pomdps.core.EnvironmentInteraction`): the transition
        """
        
        sim_result = self.simulation_step(self.state, action)
        
        reward = self.reward(self.state, action, sim_result.state)
        terminal = self.terminal(self.state, action, sim_result.state)
        
        if self._logger.isEnabledFor(LogLevel.V2.value):
            if action == self.LISTEN:
                descr = (
                    "the agent hears "
                    + self.ELEM_TO_STRING[self.obs2index(sim_result.observation)]
                )
            else:  # agent is opening door
                descr = f"the agent opens {self.ELEM_TO_STRING[action]} ({reward})"
            
            self._logger.log(
                LogLevel.V2.value,
                f"With tiger {self.ELEM_TO_STRING[self.state[0]]}, {descr}",
            )
            
        self.state = sim_result.state
        
        return DomainStepResult(sim_result.observation, reward, terminal)
    
    # 观察到index
    # to ensure the observation not out of range
    def obs2index(self, observation: np.ndarray) -> int:
        """projects the observation as an int
        Args:
             observation: (`np.ndarray`): observation to project
        RETURNS (`int`): int representation of observation
        """
        # since we have no longer use one_hot_encode_observation, then just return is okay
        assert self.observation_space.contains(
            observation
        ), f"{observation} not in space"
        
        return observation[0]
        
    
    # report what kind of encoding way, since we just use default so always default
    # but it might not be useful
    def __repr__(self) -> str:
        encoding_descr = "default"
        return f"Tiger problem ({encoding_descr} encoding) with obs prob {self._correct_obs_probs}"
