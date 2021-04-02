import logging
from collections import deque
from copy import deepcopy
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import online_pomdp_planning.types as planner_types
import pomdp_belief_tracking.types as belief_types
from general_bayes_adaptive_pomdps.baddr.model import (
    BADDr,
    DynamicsModel,
    create_transition_sampler,
    train_from_samples,
)
from general_bayes_adaptive_pomdps.baddr.neural_networks.pytorch_api import set_device
from general_bayes_adaptive_pomdps.core import Domain
from general_bayes_adaptive_pomdps.domains import create_domain, create_prior
from online_pomdp_planning.mcts import Policy
from online_pomdp_planning.mcts import create_POUCT as lib_create_POUCT
from online_pomdp_planning.mcts import create_rollout as lib_create_rollout
from pomdp_belief_tracking.pf import particle_filter as PF
from pomdp_belief_tracking.pf import rejection_sampling as RS


def main() -> None:
    """runs PO-UCT planner with a belief on BADDr"""

    logger = logging.getLogger("MABRL")

    # experiment parameters
    runs = 2
    episodes = 5
    horizon = 10
    discount = 0.95

    # planning parameters
    num_particles = 128
    num_sims = 1024
    exploration_constant = 100
    planning_horizon = 5

    # learning parameters
    optimizer = "SGD"
    num_nets = 2
    learning_rate = 0.1
    online_learning_rate = 0.01
    num_epochs = 512
    batch_size = 32
    network_size = 32

    prior_certainty = 1000
    prior_correctness = 0.1

    # TODO: improve API: give device to `BADDr`
    set_device(use_gpu=False)

    # setup
    env = create_domain(
        "tiger",
        0,
        use_one_hot_encoding=True,
    )

    baddr = BADDr(env, num_nets, optimizer, learning_rate, network_size, batch_size)

    planner = create_planner(
        baddr,
        create_rollout_policy(env),
        num_sims,
        exploration_constant,
        planning_horizon,
        discount,
    )
    belief = belief_types.Belief(
        baddr.sample_start_state, create_rejection_sampling(baddr, num_particles)
    )
    train_method = create_train_method(
        prior_certainty, prior_correctness, num_epochs, batch_size
    )

    def set_domain_state(s: BADDr.AugmentedState):
        """sets domain state in ``s`` to sampled initial state """
        return BADDr.AugmentedState(baddr.sample_domain_start_state(), s.model)

    output: List[Dict[str, Any]] = []

    for run in range(runs):

        baddr.reset(train_method, learning_rate, online_learning_rate)

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


def create_train_method(
    prior_certainty, prior_correctness, num_epochs, batch_size
) -> Callable[[DynamicsModel], None]:
    """creates a model training method

    This returns a function that can be called on any `DynamicsModel` net to be
    trained

    :param env:
    :param prior_certainty:
    :param prior_correctness:
    :param num_epochs:
    :param batch_size:
    """
    logger = logging.getLogger("train method")

    sim_sampler = create_prior(
        "tiger",
        0,
        prior_certainty,
        prior_correctness,
        use_one_hot_encoding=True,
    ).sample

    def train_method(net: DynamicsModel):
        sim = sim_sampler()
        logger.debug("Training network on %s", sim)
        sampler = create_transition_sampler(sim)
        train_from_samples(net, sampler, num_epochs, batch_size)

    return train_method


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

        if step.terminal:
            break

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
        # update the parameters of the augmented state
        copy = deepcopy(ss)
        baddr.update_theta(
            copy.model,
            ctx["state"].domain_state,
            ctx["action"],
            copy.domain_state,
            ctx["observation"],
        )
        return copy

    def belief_sim(s: BADDr.AugmentedState, a: int) -> Tuple[np.ndarray, np.ndarray]:
        out = baddr.simulation_step_without_updating_theta(s, a)
        return out.state, out.observation

    return RS.create_rejection_sampling(
        belief_sim, num_samples, np.array_equal, process_acpt  # type: ignore
    )


class RolloutPolicyForPlanning(Policy):
    """A policy for ``online_pomdp_planning`` from ``general_bayes_adaptive_pomdps`` policies"""

    def __init__(self, pol: Callable[[BADDr.AugmentedState], int]):
        """Wraps and calls ``pol`` with imposed signature

        :param pol:
        """
        super().__init__()
        self._rollout_pol = pol

    def __call__(self, s: BADDr.AugmentedState, _: np.ndarray) -> int:
        """The signature for the policy for online planning

        A stochastic mapping from state and observation to action

        :param s: the state
        :param _: the observation, ignored
        """
        return self._rollout_pol(s)


class SimForPlanning(planner_types.Simulator):
    """A simulator for ``online_pomdp_planning`` from ``general_bayes_adaptive_pomdps``"""

    def __init__(self, bnrl_simulator: BADDr):
        """Wraps and calls ``bnrl_simulator`` with imposed signature

        :param bnrl_simulator:
        """
        super().__init__()
        self._bnrl_sim = bnrl_simulator

    def __call__(
        self, s: BADDr.AugmentedState, a: int
    ) -> Tuple[np.ndarray, np.ndarray, float, bool]:
        """The signature for the simulator for online planning

        Upon calling, produces a transition (state, observation, reward, terminal)

        :param s: input state
        :param a: input action
        """
        next_s, obs = self._bnrl_sim.simulation_step_without_updating_theta(s, a)
        reward = self._bnrl_sim.reward(s, a, next_s)
        terminal = self._bnrl_sim.terminal(s, a, next_s)

        return next_s, obs.data.tobytes(), reward, terminal


def create_rollout_policy(domain: Domain) -> Policy:
    """returns a random policy

    Currently only supported by grid-verse environment:
        - "default" -- default "informed" rollout policy
        - "gridverse-extra" -- straight if possible, otherwise turn

    :param domain: true POMDP
    """

    def rollout(_: BADDr.AugmentedState) -> int:
        """
        So normally PO-UCT expects states to be numpy arrays and everything is
        dandy, but we are planning in augmented space here in secret. So the
        typical rollout policy of the environment will not work: it does not
        expect an `AugmentedState`. So here we gently provide it the underlying
        state and all is well

        :param augmented_state:
        :return: action to take during rollout
        """
        return domain.action_space.sample_as_int()

    return RolloutPolicyForPlanning(rollout)
