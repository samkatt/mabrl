import logging
from collections import deque
from functools import partial
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import online_pomdp_planning.types as planner_types
import pomdp_belief_tracking.types as belief_types
from general_bayes_adaptive_pomdps.core import ActionSpace
from general_bayes_adaptive_pomdps.domains.domain import Domain
from general_bayes_adaptive_pomdps.models.baddr import (
    BADDr,
    BADDrState,
    DynamicsModel,
    backprop_update,
    create_dynamics_model,
    sample_transitions_uniform,
    train_from_samples,
)
from online_pomdp_planning.mcts import Policy
from online_pomdp_planning.mcts import create_POUCT as lib_create_POUCT
from online_pomdp_planning.mcts import create_rollout as lib_create_rollout
from pomdp_belief_tracking.pf import particle_filter as PF
from pomdp_belief_tracking.pf import rejection_sampling as RS

from mabrl.RealDomain import RealDomain


def main() -> None:
    """runs PO-UCT planner with a belief on BADDr"""

    logger = logging.getLogger("MABRL")

    # experiment parameters
    runs = 2
    episodes = 5
    horizon = 80
    discount = 0.95

    # planning parameters
    num_particles = 128
    # num_sims = 1024
    num_sims = 2048
    # exploration_constant = 100
    exploration_constant = 200
    # planning_horizon = 5
    planning_horizon = 30

    # learning parameters
    optimizer = "SGD"
    num_nets = 2
    learning_rate = 0.1
    online_learning_rate = 0.01
    # num_epochs = 512
    num_epochs = 1024
    # batch_size = 32
    batch_size = 64
    # network_size = 32
    network_size = 64
    dropout_rate = 0.1

    model_updates = [
        partial(
            backprop_update,
            freeze_model_setting=DynamicsModel.FreezeModelSetting.FREEZE_NONE,
        )
    ]

    # setup
    env = RealDomain()

    models = [
        create_dynamics_model(
            env.state_space,
            env.action_space,
            env.observation_space,
            optimizer,
            learning_rate,
            network_size,
            batch_size,
            dropout_rate,
        )
        for _ in range(num_nets)
    ]

    sampler = partial(
        sample_transitions_uniform,
        env.state_space,
        env.action_space,
        env.simulation_step,
    )

    for model in models:
        # print out loss
        y = [0] * 1024
        for _ in range(num_epochs):
            loss = train_from_samples(
                model,
                sampler,
                num_epochs=num_epochs,
                batch_size=batch_size,
            )
            print("time: ", _, ", loss: ", loss)
            y[_] += loss
            # plt.plot(_,loss)

        # x = np.linspace(0,500,1)
        x = np.linspace(0, 1024, 1024)
        y = [i / 2 for i in y]
        plt.plot(x, y)
        plt.show()

        model.set_learning_rate(online_learning_rate)

    baddr = BADDr(
        env.action_space,
        env.observation_space,
        env.sample_start_state,
        env.reward,
        env.terminal,
        models,
        model_updates,
    )

    planner = create_planner(
        baddr,
        RandomPlicy(env.action_space),
        num_sims,
        exploration_constant,
        planning_horizon,
        discount,
    )
    belief = belief_types.Belief(
        baddr.sample_start_state, create_rejection_sampling(baddr, num_particles)
    )

    def set_domain_state(s: BADDrState):
        """sets domain state in ``s`` to sampled initial state"""
        return BADDrState(baddr.sample_domain_start_state(), s.model)

    output: List[Dict[str, Any]] = []

    for run in range(runs):

        avg_recent_return = deque([], 50)

        for episode in range(episodes):

            env.reset()

            if episode > 0:
                belief.distribution = PF.apply(
                    set_domain_state,  # type: ignore
                    belief.distribution,
                )

            episode_output = run_episode(env, planner, belief, horizon)

            # here we explicitly add the information of which run the result
            # was generated to each entry in the results
            for o in episode_output:
                o["episode"] = episode
                o["run"] = run
            output.extend(episode_output)

            discounted_return = sum(
                pow(discount, i) * o["reward"] for i, o in enumerate(episode_output)
            )
            avg_recent_return.append(discounted_return)

            logger.warning(
                "Episode %s/%s return: %s",
                episode + 1,
                episodes,
                discounted_return,
            )
            logger.info(
                f"run {run+1}/{runs} episode {episode+1}/{episodes}: "
                f"avg return: {np.mean(avg_recent_return)}"
            )


def run_episode(
    env: Domain,
    planner: planner_types.Planner,
    belief: belief_types.Belief,
    horizon: int,
) -> List[Dict[str, Any]]:
    """runs a single episode

    Returns information returned by the planner and belief in a list, where the
    nth element is the info of the nth episode step.

    Returns a list of dictionaries, one for each timestep. The dictionary
    includes things as:

        - "reward": the reward give to the agent at the time step
        - "terminal": whether the step was terminal (should really only be last, if any)
        - "timestep": the time step (should be equal to the index)
        - information from the planner info
        - information from belief info

    :param env:
    :param planner:
    :param belief:
    :param horizon: length of episode
    :return: a list of episode results (rewards and info dictionaries)
    """

    logger = logging.getLogger("episode")

    info: List[Dict[str, Any]] = []

    for timestep in range(horizon):

        # actual step
        action, planning_info = planner(belief.sample)
        assert isinstance(action, int)

        step = env.step(action)

        logger.debug("A(%s) -> O(%s) --- r(%s)", action, step.observation, step.reward)

        belief_info = belief.update(action, step.observation)

        info.append(
            {
                "timestep": timestep,
                "reward": step.reward,
                "terminal": step.terminal,
                **planning_info,
                **belief_info,
            }
        )

        # if step.terminal:
        #    break

    return info


def create_planner(
    baddr: BADDr,
    rollout_policy: Policy,
    num_sims: int = 500,
    exploration_constant: float = 1.0,
    planning_horizon: int = 10,
    discount: float = 0.95,
) -> planner_types.Planner:
    """The factory function for planners

    Currently just returns PO-UCT with given parameters, but allows for future
    generalization

    Real `env` is used for the rollout policy

    :param baddr:
    :param rollout_policy: the rollout policy
    :param num_sims: number of simulations to run
    :param exploration_constant: the UCB-constant for UCB
    :param planning_horizon: how far into the future to plan for
    :param discount: the discount factor to plan for
    """

    actions = list(i for i in range(baddr.action_space.n))
    online_planning_sim = SimForPlanning(baddr)

    rollout = lib_create_rollout(
        rollout_policy, online_planning_sim, planning_horizon, discount
    )

    return lib_create_POUCT(
        actions,
        online_planning_sim,
        num_sims,
        leaf_eval=rollout,
        discount_factor=discount,
        rollout_depth=planning_horizon,
        ucb_constant=exploration_constant,
    )


def create_rejection_sampling(
    baddr: BADDr, num_samples: int
) -> belief_types.BeliefUpdate:
    """Creates a rejection-sampling belief update

    Returns a rejection sampling belief update that tracks ``num_samples``
    particles in the ``baddr``. Basically glue between
    ``general_bayes_adaptive_pomdps`` and ``pomdp_belief_tracking``.

    Uses ``baddr`` to simulate and reject steps.

    :param baddr: the GBA-POMDP to track belief for
    :param num_samples: number of particles to main
    """

    def process_acpt(ss, ctx, _):
        return baddr.model_simulation_step(
            ss,
            ctx["state"],
            ctx["action"],
            ss,
            ctx["observation"],
        )

    def belief_sim(s: BADDrState, a: int) -> Tuple[BADDrState, np.ndarray]:
        out = baddr.domain_simulation_step(s, a)
        return out.state, out.observation

    return RS.create_rejection_sampling(
        belief_sim, num_samples, np.array_equal, process_acpt  # type: ignore
    )


class SimForPlanning(planner_types.Simulator):
    """A simulator for ``online_pomdp_planning`` from ``general_bayes_adaptive_pomdps``"""

    def __init__(self, bnrl_simulator: BADDr):
        """Wraps and calls ``bnrl_simulator`` with imposed signature

        :param bnrl_simulator:
        """
        super().__init__()
        self._bnrl_sim = bnrl_simulator

    def __call__(self, s: BADDrState, a: int) -> Tuple[np.ndarray, bytes, float, bool]:
        """The signature for the simulator for online planning

        Upon calling, produces a transition (state, observation, reward, terminal)

        :param s: input state
        :param a: input action
        """
        next_s, obs = self._bnrl_sim.domain_simulation_step(s, a)
        reward = self._bnrl_sim.reward(s, a, next_s)
        terminal = self._bnrl_sim.terminal(s, a, next_s)

        return next_s, obs.data.tobytes(), reward, terminal


class RandomPlicy:
    """A random :mod:`online_pomdp_planning.mcts.Policy` given an action space"""

    def __init__(self, A: ActionSpace):
        self.A = A

    def __call__(self, _, __) -> int:
        """Adopts interface of :class:`online_pomdp_planning.mcts.Policy`"""
        return self.A.sample_as_int()
