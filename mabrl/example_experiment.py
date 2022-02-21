import logging
from collections import deque
from typing import Any, Dict, List, Tuple

import numpy as np
import online_pomdp_planning.types as planner_types
import pomdp_belief_tracking.types as belief_types
from general_bayes_adaptive_pomdps.core import ActionSpace, GeneralBAPOMDP
from general_bayes_adaptive_pomdps.domains.domain import Domain
from general_bayes_adaptive_pomdps.domains.tiger import (
    Tiger,
    create_tabular_prior_counts,
)
from general_bayes_adaptive_pomdps.models.tabular_bapomdp import (
    TabularBAPOMDP,
    TBAPOMDPState,
)
from online_pomdp_planning.mcts import Policy
from online_pomdp_planning.mcts import create_POUCT as lib_create_POUCT
from pomdp_belief_tracking.pf import particle_filter as PF
from pomdp_belief_tracking.pf import rejection_sampling as RS


def main() -> None:
    """runs PO-UCT planner with a belief on a general BA-POMDP"""

    logger = logging.getLogger("MABRL")
    logger.setLevel(logging.DEBUG)

    # experiment parameters
    runs = 2
    episodes = 5
    horizon = 80
    discount = 0.95

    # planning parameters
    num_particles = 512
    # num_sims = 1024
    num_sims = 2048
    # exploration_constant = 100
    exploration_constant = 200
    # planning_horizon = 5
    planning_horizon = 30

    # prior configurations
    total_counts = 10
    prior_correctness = 1

    # setup
    env = Tiger(one_hot_encode_observation=False)
    prior_counts = create_tabular_prior_counts(prior_correctness, total_counts)

    tbapomdp = TabularBAPOMDP(
        env.state_space,
        env.action_space,
        env.observation_space,
        env.sample_start_state,
        env.reward,
        env.terminal,
        prior_counts,
    )

    planner = create_planner(
        tbapomdp,
        RandomPlicy(env.action_space),
        num_sims,
        exploration_constant,
        planning_horizon,
        discount,
    )
    belief = belief_types.Belief(
        tbapomdp.sample_start_state, create_rejection_sampling(tbapomdp, num_particles)
    )

    def set_domain_state(s: TBAPOMDPState):
        """sets domain state in ``s`` to sampled initial state"""
        return TBAPOMDPState(tbapomdp.sample_domain_start_state(), s.counts)

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
                "Episode %d/%d return: %f.2",
                episode + 1,
                episodes,
                discounted_return,
            )
            logger.info(
                "run %d/%d episode %d/%d: avg return: %f.2",
                run + 1,
                runs,
                episode + 1,
                episodes,
                np.mean(avg_recent_return),
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

        logger.warn("A(%s) -> O(%s) --- r(%s)", action, step.observation, step.reward)

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

        if step.terminal:
            break

    return info


def create_planner(
    gba_pomdp: GeneralBAPOMDP,
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

    :param gba_pomdp:
    :param rollout_policy: the rollout policy
    :param num_sims: number of simulations to run
    :param exploration_constant: the UCB-constant for UCB
    :param planning_horizon: how far into the future to plan for
    :param discount: the discount factor to plan for
    """

    actions = list(i for i in range(gba_pomdp.action_space.n))
    online_planning_sim = SimForPlanning(gba_pomdp)

    return lib_create_POUCT(
        actions,
        online_planning_sim,
        num_sims,
        discount_factor=discount,
        rollout_depth=planning_horizon,
        ucb_constant=exploration_constant,
    )


def create_rejection_sampling(
    gba_pomdp: GeneralBAPOMDP, num_samples: int
) -> belief_types.BeliefUpdate:
    """Creates a rejection-sampling belief update

    Returns a rejection sampling belief update that tracks ``num_samples``
    particles in the ``gba_pomdp``. Basically glue between
    ``general_bayes_adaptive_pomdps`` and ``pomdp_belief_tracking``.

    Uses ``gba_pomdp`` to simulate and reject steps.

    :param gba_pomdp: the GBA-POMDP to track belief for
    :param num_samples: number of particles to main
    """

    def process_acpt(ss, ctx, _):
        return gba_pomdp.model_simulation_step(
            ss,
            ctx["state"],
            ctx["action"],
            ss,
            ctx["observation"],
        )

    def belief_sim(s: TBAPOMDPState, a: int) -> Tuple[TBAPOMDPState, np.ndarray]:
        out = gba_pomdp.domain_simulation_step(s, a)
        return out.state, out.observation

    return RS.create_rejection_sampling(
        belief_sim, num_samples, np.array_equal, process_acpt  # type: ignore
    )


class SimForPlanning(planner_types.Simulator):
    """A simulator for ``online_pomdp_planning`` from ``general_bayes_adaptive_pomdps``"""

    def __init__(self, bnrl_simulator: GeneralBAPOMDP):
        """Wraps and calls ``bnrl_simulator`` with imposed signature

        :param bnrl_simulator:
        """
        super().__init__()
        self._bnrl_sim = bnrl_simulator

    def __call__(self, s, a: int) -> Tuple[np.ndarray, bytes, float, bool]:
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
