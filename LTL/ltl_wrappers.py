"""
This is a simple wrapper that will include LTL goals to any given environment.
It also progress the formulas as the agent interacts with the envirionment.

However, each environment must implement the followng functions:
    - *get_events(...)*: Returns the propositions that currently hold on the environment.
    - *get_propositions(...)*: Maps the objects in the environment to a set of
                            propositions that can be referred to in LTL.

Notes about LTLEnv:
    - The episode ends if the LTL goal is progressed to True or False.
    - If the LTL goal becomes True, then an extra +1 reward is given to the agent.
    - If the LTL goal becomes False, then an extra -1 reward is given to the agent.
    - Otherwise, the agent gets the same reward given by the original environment.
"""
import torch

import numpy as np
import gym
from gym import spaces
import LTL.ltl_progression, random
# from ltl_samplers import getLTLSampler, SequenceSampler

def prYellow(prt): print("\033[93m {}\033[00m".format(prt))

class LTLEnv(gym.Wrapper):
    def __init__(self, env, progression_mode="full", intrinsic=0.0, use_Waypoint=False, bounds=None, task_name=None, way_weight=1., way_dist=0.1):
        """
        LTL environment
        --------------------
        It adds an LTL objective to the current environment
            - The observations become a dictionary with an added "text" field
              specifying the LTL objective
            - It also automatically progress the formula and generates an
              appropriate reward function
            - However, it does requires the user to define a labeling function
              and a set of training formulas
        progression_mode:
            - "full": the agent gets the full, progressed LTL formula as part of the observation
            - "partial": the agent sees which propositions (individually) will progress or falsify the formula
            - "none": the agent gets the full, original LTL formula as part of the observation
        """
        super().__init__(env)
        self.progression_mode   = progression_mode
        # self.propositions = self.env.get_propositions()  # ['J', 'W', 'R', 'Y']

        self.observation_space = spaces.Dict({'features': env.observation_space})
        self.known_progressions = {}
        self.intrinsic = intrinsic

        self.task_name = task_name

        # fixme: waypoint
        self.use_waypoint = use_Waypoint
        if self.use_waypoint:
            self.way_weight = way_weight
            self.way_dist = way_dist
            self.goal_indx = 0
            self._stop = False
            self.bounds = bounds


    def sample_ltl_goal(self):
        # This function must return an LTL formula for the task
        # Format:
        #(
        #    'and',
        #    ('until','True', ('and', 'd', ('until','True',('not','c')))),
        #    ('until','True', ('and', 'a', ('until','True', ('and', 'b', ('until','True','c')))))
        #)
        # NOTE: The propositions must be represented by a char
        raise NotImplementedError

    def get_events(self, obs, act, next_obs):
        # This function must return the events that currently hold on the environment
        # NOTE: The events are represented by a string containing the propositions with positive values only (e.g., "ac" means that only propositions 'a' and 'b' hold)
        raise NotImplementedError

    def reset(self):
        self.known_progressions = {}
        self.obs = self.env.reset()

        # Defining an LTL goal
        # self.ltl_goal = self.sample_ltl_goal()
        # self.ltl_goal = ('and', ('eventually', 'push'), ('eventually', 'pickandplace'))
        if self.task_name == 'Cleanup':
            self.ltl_goal     = ('and', ('eventually', 'push'), ('eventually', 'pickandplace'))
        elif self.task_name == 'NutAssemblyRound':
            self.ltl_goal = ('eventually', ('and', 'reached', ('eventually', ('and', 'grasped', ('eventually', 'nuted')))))
        elif self.task_name == 'Stack':
            self.ltl_goal = ('eventually', ('and', 'reach_grasp',('eventually', 'stacked')))
        elif self.task_name == 'Lift':
            self.ltl_goal = ('eventually', ('and', 'grasped',('eventually', 'lifted')))
        elif self.task_name == 'PegInHole':
            self.ltl_goal = ('eventually', ('and', 'grasped', ('eventually', ('and', 'aligned', ('eventually', 'inserted')))))
        else:
            raise ValueError(f"Unknown task name: {self.task_name}")

        # else:
        #     raise
        self.ltl_original = self.ltl_goal

        # Adding the ltl goal to the observation
        if self.progression_mode == "partial":
            ltl_obs = {'features': self.obs,'progress_info': self.progress_info(self.ltl_goal)}
        else:
            ltl_obs = {'features': self.obs,'text': self.ltl_goal}

        # fixme: waypoint
        self.goal_indx = 0
        self._stop = False

        return ltl_obs


    def step(self, action, image_obs_in_info):
        int_reward = 0.0
        # executing the action in the environment
        next_obs, original_reward, env_done, info = self.env.step(action, image_obs_in_info=image_obs_in_info)
        info.update(dict(r_env_total=original_reward))

        if self.use_waypoint:
            reach = self.reach(next_obs[0:3])
            if (self.goal_indx == 0 or reach) and self.goal_indx < len(self.waypoints):
                self.current_goal = self.waypoints[self.goal_indx]
                self.goal_indx += 1
            # obs[:3] = obs[:3] - self.current_goal
            next_obs = np.concatenate([next_obs, self.current_goal], axis=-1)
            # reach all goals - task complete
            if self.goal_indx == len(self.waypoints):
                if not env_done:
                    # prYellow(f"reach step: {self.step_nums}")
                    self.stop_in_next_step()
            if reach:
                ctrl_r = 1.
            else:
                ctrl_r = np.linalg.norm(self.current_goal-self.obs[:3]) - np.linalg.norm(next_obs[:3]-self.current_goal)
            original_reward += self.way_weight * ctrl_r  # todo: add to info['r_waypoint']


        if self.task_name == 'Cleanup':
            self.pnp_success = info['success_pnp']
            self.push_success = info['success_push']
            self.all_success = info['success']

            # progressing the ltl formula
            if self.pnp_success:
                truth_assignment = 'pickandplace'
            elif self.push_success:
                truth_assignment = 'push'
            else:
                truth_assignment = ''
        elif self.task_name == 'NutAssemblyRound':
            self.reach_success = info['r_reach']
            self.grasp_success = info['r_grasp']
            self.nut_success = info['success']

            # progressing the ltl formula
            if self.reach_success == 1.0:
                truth_assignment = 'reached'
            elif self.grasp_success == 1.0:
                truth_assignment = 'grasped'
            elif self.nut_success:
                truth_assignment = 'nuted'
            else:
                truth_assignment = ''
        elif self.task_name == 'Stack':
            self.reach_grasp_success = info['r_reach_grasp']
            self.stack_success = info['success']
            # progressing the ltl formula
            if self.reach_grasp_success == 1.0:
                truth_assignment = 'reach_grasp'
            elif self.stack_success:
                truth_assignment = 'stacked'
            else:
                truth_assignment = ''
        elif self.task_name == 'Lift':
            self.grasp_success = info['grasped']
            self.lift_success = info['success']

            # progressing the ltl formula
            if self.grasp_success == 1.0:
                truth_assignment = 'grasped'
            elif self.lift_success == 1.0:
                truth_assignment = 'lifted'
            else:
                truth_assignment = ''
        elif self.task_name == 'PegInHole':
            self.grasp_success = info['r_grasp']
            self.align_success = info['r_align']
            self.insert_success = info['success']

            # progressing the ltl formula
            if self.grasp_success > 0.:
                truth_assignment = 'grasped'
            elif self.align_success > 0.:
                truth_assignment = 'aligned'
            elif self.insert_success:
                truth_assignment = 'inserted'
            else:
                truth_assignment = ''
        else:
            raise ValueError(f"Unknown task name: {self.task_name}")


        self.ltl_goal = self.progression(self.ltl_goal, truth_assignment)
        self.obs      = next_obs

        # Computing the LTL reward and done signal

        ltl_done   = False
        if self.ltl_goal == 'True':
            ltl_done   = True
            ltl_reward = 1.
        elif self.ltl_goal == 'False':
            ltl_done   = True
            ltl_reward = -1.
        else:
            ltl_reward = int_reward

        # Computing the new observation and returning the outcome of this action
        if self.progression_mode == "full":
            ltl_obs = {'features': self.obs,'text': self.ltl_goal}
        elif self.progression_mode == "none":
            ltl_obs = {'features': self.obs,'text': self.ltl_original}
        elif self.progression_mode == "partial":
            ltl_obs = {'features': self.obs, 'progress_info': self.progress_info(self.ltl_goal)}
        else:
            raise NotImplementedError

        if self.use_waypoint:
            info.update(dict(ctrl_r=self.way_weight * ctrl_r + ltl_reward))

        reward  = original_reward + ltl_reward
        done    = env_done or ltl_done
        return ltl_obs, reward, done, info

    def progression(self, ltl_formula, truth_assignment):

        if (ltl_formula, truth_assignment) not in self.known_progressions:
            result_ltl = LTL.ltl_progression.progress_and_clean(ltl_formula, truth_assignment)
            self.known_progressions[(ltl_formula, truth_assignment)] = result_ltl

        return self.known_progressions[(ltl_formula, truth_assignment)]


    # # X is a vector where index i is 1 if prop i progresses the formula, -1 if it falsifies it, 0 otherwise.
    def progress_info(self, ltl_formula):
        propositions = self.env.get_propositions()
        X = np.zeros(len(self.propositions))

        for i in range(len(propositions)):
            progress_i = self.progression(ltl_formula, propositions[i])
            if progress_i == 'False':
                X[i] = -1.
            elif progress_i != ltl_formula:
                X[i] = 1.
        return X

    def sample_ltl_goal(self):
        # NOTE: The propositions must be represented by a char
        # This function must return an LTL formula for the task
        formula = self.sampler.sample()

        if isinstance(self.sampler, SequenceSampler):
            def flatten(bla):
                output = []
                for item in bla:
                    output += flatten(item) if isinstance(item, tuple) else [item]
                return output

            length = flatten(formula).count("and") + 1
            self.env.timeout = 25 # 10 * length

        return formula


    def get_events(self, obs, act, next_obs):
        # This function must return the events that currently hold on the environment
        # NOTE: The events are represented by a string containing the propositions with positive values only (e.g., "ac" means that only propositions 'a' and 'b' hold)
        return self.env.get_events()

    def set_waypoints(self, waypoints):
        if self.bounds is not None:
            self.waypoints = self._get_unnormalized_pos(waypoints)
        else:
            self.waypoints = waypoints
        self.current_goal = self.waypoints[self.goal_indx]
        self.goal_indx += 1
        self._stop = False
        obs = np.concatenate([self.obs, self.current_goal], axis=-1)
        ltl_obs = {'features': obs, 'text': self.ltl_goal}
        return ltl_obs

    def reach(self, obs):
        return np.linalg.norm(obs - self.current_goal) < self.way_dist

    def stop_in_next_step(self):
        self._stop = True

    def _get_unnormalized_pos(self, pos):
        pos = np.clip(pos, -1, 1)
        pos = (pos + 1) / 2
        # low, high = bounds[0], bounds[1]
        low, high = np.array(self.bounds[0]), np.array(self.bounds[1])
        return low + (high - low) * pos


class NoLTLWrapper(gym.Wrapper):
    def __init__(self, env, use_Waypoint=False, bounds=None, task_name=None, way_weight=1., way_dist=0.1):
        """
        Removes the LTL formula from an LTLEnv
        It is useful to check the performance of off-the-shelf agents
        """
        super().__init__(env)
        self.observation_space = env.observation_space

        self.task_name = task_name

        self.use_waypoint = use_Waypoint
        if self.use_waypoint:
            self.way_weight = way_weight
            self.way_dist = way_dist
            self.goal_indx = 0
            self._stop = False
            self.bounds = bounds
        # self.observation_space =  env.observation_space['features']

    def reset(self):
        self.obs = self.env.reset()
        self.goal_indx = 0
        self._stop = False
        # obs = obs['features']
        # obs = {'features': obs}
        return self.obs

    def step(self, action, image_obs_in_info):
        # executing the action in the environment
        obs, reward, done, info = self.env.step(action, image_obs_in_info=image_obs_in_info)
        info.update(dict(r_env_total=reward))
        if self.use_waypoint:
            reach = self.reach(obs[0:3])
            if (self.goal_indx == 0 or reach) and self.goal_indx < len(self.waypoints):
                self.current_goal = self.waypoints[self.goal_indx]
                self.goal_indx += 1
            # obs[:3] = obs[:3] - self.current_goal
            obs = np.concatenate([obs, self.current_goal], axis=-1)
            # reach all goals - task complete
            if self.goal_indx == len(self.waypoints):
                if not done:
                    # prYellow(f"reach step: {self.step_nums}")
                    self.stop_in_next_step()
            if reach:
                ctrl_r = 1.
            else:
                ctrl_r = np.linalg.norm(self.current_goal-self.obs[:3]) - np.linalg.norm(obs[:3]-self.current_goal)
            reward += self.way_weight * ctrl_r  # todo: add to info['r_waypoint']
        done = done or self._stop
        self.obs = obs
        return obs, reward, done, info

    def set_waypoints(self, waypoints):
        if self.bounds is not None:
            self.waypoints = self._get_unnormalized_pos(waypoints)
        else:
            self.waypoints = waypoints
        self.current_goal = self.waypoints[self.goal_indx]
        self.goal_indx += 1
        self._stop = False
        return np.concatenate([self.obs, self.current_goal], axis=-1)

    def get_propositions(self):
        return list([])

    def reach(self, obs):
        return np.linalg.norm(obs - self.current_goal) < self.way_dist

    def stop_in_next_step(self):
        self._stop = True

    def _get_unnormalized_pos(self, pos):
        pos = np.clip(pos, -1, 1)
        pos = (pos + 1) / 2
        # low, high = bounds[0], bounds[1]
        low, high = np.array(self.bounds[0]), np.array(self.bounds[1])
        return low + (high - low) * pos

if __name__ == '__main__':
    known_progressions = {}
    # ltl_goal = ('and', ('eventually', ('and', 'i', ('eventually', 'e'))),
    #             ('eventually', ('and', 'k', ('eventually', ('and', ('or', 'b', 'l'), ('eventually', ('and', 'h', ('eventually', ('and', 'e', ('eventually', 'f'))))))))))
    ltl_goal = ('and', ('eventually', 'a'), ('eventually', 'b'))
    ltl_original = ltl_goal
    truth_assignment = ['b', 'a']
    for i in range(len(truth_assignment)):
        if (ltl_goal, truth_assignment[i]) not in known_progressions:
            result_ltl = LTL.ltl_progression.progress_and_clean(ltl_goal, truth_assignment[i])
            known_progressions[(ltl_goal, truth_assignment[i])] = result_ltl
            prYellow(result_ltl)
            ltl_goal = result_ltl