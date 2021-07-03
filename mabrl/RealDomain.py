"""The tiger problem implemented as domain"""

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

# 改造成三门
class RealDomain(domain.Domain):
    """The actual domain"""
    # the rewards while human status completed
    GOOD_DOOR_REWARD = 10
    # the initial state reward equals to 0
    reward = 0
    # We don't use one hot encode observation here
    # we set the termianl step here
    terminal_step = 150
    LISTEN = 3
    # robot location
    ELEM_TO_STRING = ["on the table", "with the robot", "delivered to the workroom"]
    # robot action
    ACTION_TO_STRING = ["Go-to-WorkRoom","Go-to-ToolRoom","Pick-up","Listen", "Drop"]
    
    # might add sth else
    LISTEN_REWARD = -1
    # We can thinking adding one human function to think about it.
    
    def __init__(
        self,
        # one_hot_encode_observation: bool,
        correct_obs_probs: Optional[List[float]] = None,
    ):
        """Construct the tiger domain
        Args:
             one_hot_encode_observation: (`bool`):
             correct_obs_probs: (`Optional[List[float]]`):
        """
        # 可以假设正确的概率全部为0.85
        # observation: human working status;
        
        if not correct_obs_probs:
            correct_obs_probs = [0.85]
            
        # to ensure the probabilty in normal range
        assert (
            0 <= correct_obs_probs[0] <= 1
        ), f"observation prob {correct_obs_probs[0]} not a probability"
    
        
        # Logger 用法
        self._logger = Logger(self.__class__.__name__)
        
        self._correct_obs_probs = correct_obs_probs
        
        # self._use_one_hot_obs = one_hot_encode_observation
        
        # state 包含机器人位置 [0,1], 人类工作状态 0 1 2 3 4, 四个工具的状态 1 2 3 4分别赋值为0 1 2 means 桌子上, 机器人上，送到了
        # state includes these feature with different values
        # 1. robot location [0: tool room, 1: work room]
        # 2. human working status: 0 1 2 3 4
        # 3. four tools' status: 0 -> on the table, 1-> with robot, 2-> delivered(drop)
        self._state_space = DiscreteSpace([2,5,3,3,3,3])

        # action在此处domain有四个动作，去workroom，去tool room，捡拾工具，listen
        # there are four actions, go-to-workroom, go-to-toolroom, pick-up, listen
        # if we set listen values as 4 or 5, then error happens. I think if we want to add more acitons, then we have to keep LISTEN == 3
        self._action_space = ActionSpace(5)
        # 查明encode observation的作用
        # 此处仅仅观察human working status
        # human status has 5 values 0 1 2 3 4, plus one null value should be 6
        # 但是包含None,故空间大小为2
        # only observe the human workig status
        self._obs_space = DiscreteSpace([6])
        # 采样出事状态
        self._state = self.sample_start_state()
        
    #可读
    @property
    def state(self):
        """ returns current state """
        return self._state
    
    # state.setter
    # 转换为setter:可写
    @state.setter
    def state(self, state: np.ndarray):
        """sets state
        Args:
             state: (`np.ndarray`): [0] or [1]
        """
        # to ensure something
        # state[0]应该只有一个元素但是总共3种可能即0，1，2
        assert state.shape == (6,), f"{state} not correct shape"
        # location feature
        assert 2 > state[0] >= 0, f"{state} not valid"
        # human workig status feature:0 1 2 3 4
        assert 5 > state[1] >= 0, f"{state} not valid"
        # four tool's status 0: tool room, 1: with robot 2: delivered
        for i in range(2,6):
            assert 3 > state[i] >= 0, f"{state} not valid"
            
        print("initial state: ",state)
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
        # print(observation)
        return np.array([observation],dtype=int)
    
    # 静态方法
    @staticmethod
    def sample_start_state() -> np.ndarray:
        """samples a random state (tiger left or right)
        RETURNS (`np.narray`): an initial state (in [[0],[1]])
        """
        #
        return np.array([0,0,0,0,0,0], dtype=int)
    
    #采样observation
    def sample_observation(self, loc: int, listening: bool) -> int:
        """samples an observation, listening stores whether agent is listening
        Args:
             loc: (`int`): 0 is tiger left, 1 is tiger right
             listening: (`bool`): whether the agent is listening
        RETURNS (`int`): the observation: 0 = left, 1 = right, 2 = null
        """
        # huamn working status: 0 1 2 3 4 + Null
        if not listening:
            # string, bytes like or a number
            return False
        if loc == 4:
            return loc
        else:
            return (loc if np.random.random() < self._correct_obs_probs[0] else np.random.randint(loc+1,5))
    
    def reset(self) -> np.ndarray:
        """Resets internal state and return first observation
        Resets the internal state randomly ([0] or [1])
        Returns [1,1] as a 'null' initial observation
        """
        self._state = self.sample_start_state()
        # observation 存在null
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
        # here action有4个值，0 1 2 3 对应的是去tool room, 去workroom，拾起工具，及观察
        # during the time, the human has status to complete the work.
        new_state = state.copy()
        # print("new state: ",new_state)
        # only when agent at workroom, then the observation can be done
        if action == self.LISTEN and state[0] == 1:
            obs = self.sample_observation(state[1], True)
        else:  # not opening door
            obs = self.sample_observation(state[1], False)
            
        # to illustrate the effect of the action to the new state
        # [0 1 2 2 2 2]
        if action == 0:
            new_state[0] = 0
        if action == 1:
            new_state[0] = 1
        if action == 2:
            if state[0] == 0:
                for i in range(2, 6):
                # only pick one tool a time
                    if new_state[i] == 0:
                        new_state[i] = 1
                        break
            else:
                for i in range(2, 6):
                # only pick one tool a time
                    if new_state[i] == 2:
                        new_state[i] = 1
                        break
        
        # drop
        if action == 4:
            if state[0] == 1:
                for i in range(2, 6):
                # only drop one tool a time
                    if new_state[i] == 1:
                        new_state[i] = 2
                        break
            else:
                for i in range(2, 6):
                # only drop one tool a time
                    if new_state[i] == 1:
                        new_state[i] = 0
                        break
        
        
            
        # here we can set a probability to change the human's current working status
        # 0 1 2 3 4 -> loc + status + [1 2 3 4]
        a = new_state[1]
        if new_state[1] < 4 and new_state[a+2] == 2:
            new_state[1] += 1
            a += 1

        
        return SimulationResult(new_state, self.encode_observation(obs))
    
# reward
# here for the reward, we just focus on the human working status changes and the time interval between changes
# my episodes is to reduce some interval values as the punishment
    
    def reward(self, state: np.ndarray, action: int, new_state: np.ndarray) -> float:
        """A constant if listening, penalty if opening to door, and reward otherwise
        Args:
             state: (`np.ndarray`):
             action: (`int`):
             new_state: (`np.ndarray`):
        RETURNS (`float`):
        """
        #print("old state: ",state)
        # here we need to focus on the change of the human status between new state and old state
        # we just need to focus on the human assembling task this time.
        assert self.state_space.contains(state), f"{state} not in space"
        assert self.action_space.contains(action), f"{action} not in space"
        assert self.state_space.contains(new_state), f"{new_state} not in space"
        
        # Or we can Just add something like gaussian sampling here
        # then there are values < 0 appears
        #print("new state: ",new_state)
        return (self.GOOD_DOOR_REWARD) * (new_state[1]-state[1])
    
    # 中止指令
    def terminal(self, state: np.ndarray, action: int, new_state: np.ndarray) -> bool:
        """True if opening a door
        Args:
             state: (`np.ndarray`):
             action: (`int`):
             new_state: (`np.ndarray`):
        RETURNS (`bool`):
        """
        # 确保状态空间包含原有状态与新状态
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
        # only observation not false, then we can present what it hears
        if self._logger.isEnabledFor(LogLevel.V2.value):
            if action == self.LISTEN and sim_result.observation != False:
                descr = (
                    "the agent hears "
                    + self.ELEM_TO_STRING[self.obs2index(sim_result.observation)]
                )
            else:  # agent is opening door
                descr = f"the agent takes {self.ACTION_TO_STRING[action]} ({reward})"
            
            self._logger.log(
                LogLevel.V2.value,
                f"With agent {self.ELEM_TO_STRING[self.state[0]]}, {descr}",
            )
            
        self.state = sim_result.state
        
        return DomainStepResult(sim_result.observation, reward, terminal)
    
    # 观察到index, since it don't use one_hot_encoding then we just return the observation
    def obs2index(self, observation: np.ndarray) -> np.ndarray:
        """projects the observation as an int
        Args:
             observation: (`np.ndarray`): observation to project
        RETURNS (`int`): int representation of observation
        """
        
        assert self.observation_space.contains(
            observation
        ), f"{observation} not in space"
        
        return observation

    
    def __repr__(self) -> str:
        encoding_descr = "default"
        return f"Real domain problem ({encoding_descr} encoding) with obs prob {self._correct_obs_probs}"
    
    
    
    