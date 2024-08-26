from functools import partial

import torch
from torch import Tensor

import numpy as np
import copy

create_rollout_function = partial


def multitask_rollout(
        env,
        agent,
        max_path_length=np.inf,
        render=False,
        render_kwargs=None,
        observation_key=None,
        desired_goal_key=None,
        get_action_kwargs=None,
        return_dict_obs=False,
        full_o_postprocess_func=None,
):
    if full_o_postprocess_func:
        def wrapped_fun(env, agent, o):
            full_o_postprocess_func(env, agent, observation_key, o)
    else:
        wrapped_fun = None

    def obs_processor(o):
        return np.hstack((o[observation_key], o[desired_goal_key]))

    paths = rollout(
        env,
        agent,
        max_path_length=max_path_length,
        render=render,
        render_kwargs=render_kwargs,
        get_action_kwargs=get_action_kwargs,
        preprocess_obs_for_policy_fn=obs_processor,
        full_o_postprocess_func=wrapped_fun,
    )
    if not return_dict_obs:
        paths['observations'] = paths['observations'][observation_key]
    return paths


def contextual_rollout(
        env,
        agent,
        observation_key=None,
        context_keys_for_policy=None,
        obs_processor=None,
        **kwargs
):
    if context_keys_for_policy is None:
        context_keys_for_policy = ['context']

    if not obs_processor:
        def obs_processor(o):
            combined_obs = [o[observation_key]]
            for k in context_keys_for_policy:
                combined_obs.append(o[k])
            return np.concatenate(combined_obs, axis=0)
    paths = rollout(
        env,
        agent,
        preprocess_obs_for_policy_fn=obs_processor,
        **kwargs
    )
    return paths


def rollout(
        gnn,
        preprocess_obss,
        use_LTL,
        use_waypoint,
        plan_layer,
        env,
        agent,
        max_path_length=np.inf,
        render=False,
        render_kwargs=None,
        preprocess_obs_for_policy_fn=None,
        get_action_kwargs=None,
        return_dict_obs=False,
        full_o_postprocess_func=None,
        reset_callback=None,
        addl_info_func=None,
        image_obs_in_info=False,
        last_step_is_terminal=False,
        terminals_all_false=False,
):
    if render_kwargs is None:
        render_kwargs = {}
    if get_action_kwargs is None:
        get_action_kwargs = {}
    if preprocess_obs_for_policy_fn is None:
        preprocess_obs_for_policy_fn = lambda env, agent, o: o
    raw_obs = []
    raw_next_obs = []
    observations = []
    actions = []
    rewards = []
    terminals = []
    agent_infos = []
    env_infos = []
    next_observations = []
    addl_infos = []
    path_length = 0
    agent.reset()
    o = env.reset()
    if use_waypoint:
        ctrl_r = []
        if use_LTL:
            planned_path_tensor = plan_layer.policy.forward(torch.tensor(o['features'][:3], dtype=torch.float32, device="cuda"))[0]
        else:
            planned_path_tensor = plan_layer.policy.forward(torch.tensor(o[:3], dtype=torch.float32, device="cuda"))[0]
        planned_path = as_numpy(planned_path_tensor)
        # add planner policy predict results to buffer
        plan_layer.buffer.add_traj(planned_path)
        plan_layer.buffer.add_log_prob(plan_layer.policy.log_prob(planned_path_tensor).detach())
        o = env.set_waypoints(planned_path)
    if reset_callback:
        reset_callback(env, agent, o)
    if render:
        env.render(**render_kwargs)
    while path_length < max_path_length:
        raw_obs.append(o)
        o_for_agent = preprocess_obs_for_policy_fn(env, agent, o)
        preprocessed_o = preprocess_obss([o_for_agent])
        if use_LTL:
            embedding_ltl = gnn(preprocessed_o.text)
        else:
            embedding_ltl = None
        embedding_state = torch.cat((preprocessed_o.image, embedding_ltl), dim=1) if embedding_ltl is not None else preprocessed_o.image
        a, agent_info = agent.get_action(embedding_state.squeeze().cuda().detach(), **get_action_kwargs)

        if full_o_postprocess_func:
            full_o_postprocess_func(env, agent, o)

        if addl_info_func:
            addl_infos.append(addl_info_func(env, agent, o, a))

        next_o, r, d, env_info = env.step(copy.deepcopy(a), image_obs_in_info=image_obs_in_info)

        new_path_length = path_length + env_info.get('num_ac_calls', 1)

        if new_path_length > max_path_length:
            break
        path_length = new_path_length

        if render:
            env.render(**render_kwargs)
        # observations.append(o)
        observations.append(embedding_state.squeeze().detach().numpy())  # todo: check the influence to effect
        rewards.append(r)
        if use_waypoint:
            ctrl_r.append(env_info["ctrl_r"])
        if terminals_all_false:
            terminals.append(False)
        else:
            terminals.append(d)
        actions.append(a)
        preprocessed_next_o = preprocess_obss([next_o])
        if use_LTL:
            embedding_nexrt_state = torch.cat((preprocessed_next_o.image, gnn(preprocessed_next_o.text)), dim=1)
        else:
            embedding_nexrt_state = preprocessed_next_o.image
        next_observations.append(embedding_nexrt_state.squeeze().detach().numpy())
        raw_next_obs.append(next_o)
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        if d:
            break
        o = next_o
    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    observations = np.array(observations)
    next_observations = np.array(next_observations)
    if return_dict_obs:
        observations = raw_obs
        next_observations = raw_next_obs
    rewards = np.array(rewards)
    if len(rewards.shape) == 1:
        rewards = rewards.reshape(-1, 1)

    if use_waypoint:
        ctrl_rs = np.array(ctrl_r)
        if len(ctrl_rs.shape) == 1:
            ctrl_rs = ctrl_rs.reshape(-1, 1)
        plan_layer.buffer.add_ctrl_reward(np.sum(ctrl_rs))

    path_length_actions = np.sum(
        [info.get('num_ac_calls', 1) for info in env_infos]
    )

    reward_actions_sum = np.sum(
        [info.get('reward_actions', 0) for info in env_infos]
    )

    if last_step_is_terminal:
        terminals[-1] = True

    skill_names = []
    sc = env.env.skill_controller
    for i in range(len(actions)):
        ac = actions[i]
        skill_name = sc.get_skill_name_from_action(ac)
        skill_names.append(skill_name)
        success = env_infos[i].get('success', False)
        if success:
            break

    return dict(
        observations=observations,
        actions=actions,
        rewards=rewards,
        next_observations=next_observations,
        terminals=np.array(terminals).reshape(-1, 1),
        agent_infos=agent_infos,
        env_infos=env_infos,
        addl_infos=addl_infos,
        full_observations=raw_obs,
        full_next_observations=raw_obs,
        path_length=path_length,
        path_length_actions=path_length_actions,
        reward_actions_sum=reward_actions_sum,
        skill_names=skill_names,
        max_path_length=max_path_length,
    )


def deprecated_rollout(
        env,
        agent,
        max_path_length=np.inf,
        render=False,
        render_kwargs=None,
):
    """
    The following value for the following keys will be a 2D array, with the
    first dimension corresponding to the time dimension.
     - observations
     - actions
     - rewards
     - next_observations
     - terminals

    The next two elements will be lists of dictionaries, with the index into
    the list being the index into the time
     - agent_infos
     - env_infos
    """
    if render_kwargs is None:
        render_kwargs = {}
    observations = []
    actions = []
    rewards = []
    terminals = []
    agent_infos = []
    env_infos = []
    o = env.reset()
    agent.reset()
    next_o = None
    path_length = 0
    if render:
        env.render(**render_kwargs)
    while path_length < max_path_length:
        a, agent_info = agent.get_action(o)
        next_o, r, d, env_info = env.step(a)
        observations.append(o)
        rewards.append(r)
        terminals.append(d)
        actions.append(a)
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        path_length += 1
        if d:
            break
        o = next_o
        if render:
            env.render(**render_kwargs)

    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    observations = np.array(observations)
    if len(observations.shape) == 1:
        observations = np.expand_dims(observations, 1)
        next_o = np.array([next_o])
    next_observations = np.vstack(
        (
            observations[1:, :],
            np.expand_dims(next_o, 0)
        )
    )
    return dict(
        observations=observations,
        actions=actions,
        rewards=np.array(rewards).reshape(-1, 1),
        next_observations=next_observations,
        terminals=np.array(terminals).reshape(-1, 1),
        agent_infos=agent_infos,
        env_infos=env_infos,
    )


def as_numpy(inpt) -> np.ndarray:
    if isinstance(inpt, Tensor):
        return inpt.detach().cpu().numpy()
    else:
        return np.array(inpt)