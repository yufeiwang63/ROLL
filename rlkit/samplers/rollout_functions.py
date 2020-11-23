import numpy as np
import cv2


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
):
    if render_kwargs is None:
        render_kwargs = {}
    if get_action_kwargs is None:
        get_action_kwargs = {}
    dict_obs = []
    dict_next_obs = []
    observations = []
    actions = []
    rewards = []
    terminals = []
    agent_infos = []
    env_infos = []
    next_observations = []
    path_length = 0
    agent.reset()
    o = env.reset()

    if render:
        env.render(**render_kwargs)
    goal = o[desired_goal_key]
    # print("Max path length: ", max_path_length )
    while path_length < max_path_length:
        dict_obs.append(o)
        if observation_key:
            o = o[observation_key]
        # print("in rollout, o.shape: ", o.shape)
        # print("in rollout, goal.shape: ", goal.shape)
        new_obs = np.hstack((o, goal))
        a, agent_info = agent.get_action(new_obs, **get_action_kwargs)
        next_o, r, d, env_info = env.step(a)
        #print("Reward from env shape:", r.shape)
        #print("Reward type from env:", type(r))
        #print("Reward: ", r)

        #print("next_o shape", next_o.shape)
        #print("next_o keys", next_o.keys())
        #print(next_o['achieved_goal'])
        #print(next_o['state_observation'])
        #print(next_o['state_desired_goal'])
        #print(next_o['state_achieved_goal'])
        #print(next_o['latent_achieved_goal'])
        #print(next_o['latent_desired_goal'])
        #print(next_o['desired_goal'])
        #print(next_o['latent_observation'])
        #print()
        #print(next_o['image_observation'].shape)
        #print(next_o['observation'].shape)
        #print(next_o['image_desired_goal'].shape)
        #print(next_o['image_achieved_goal'].shape)
        #print(next_o['state_observation'].shape)

        #imsize = 48
        #img = next_o["image_observation"]
        #img = img.reshape(3, imsize, imsize).transpose()
        #img = img[::-1, :, ::-1]
        #img *= 255
        ##img = np.reshape(img, [48,48,3])
        #cv2.imshow("img", img)
        #cv2.waitKey()
        
        if render:
            env.render(**render_kwargs)
        observations.append(o)
        rewards.append(r)
        terminals.append(d)
        actions.append(a)
        next_observations.append(next_o)
        dict_next_obs.append(next_o)
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        path_length += 1
        if d:
            break
        o = next_o
    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    observations = np.array(observations)
    next_observations = np.array(next_observations)

    #print("Observations shape", observations.shape)
    #print("Next observations shape", next_observations.shape)
    #print("Next observations shape[0]", next_observations[0].shape)
    #print("Rewards for a path shape:", len(rewards))
    #print("Rewards for a path shape 0:", rewards[0].shape)
    #print("Rewards 0:", rewards[0])

    
    if return_dict_obs:
        observations = dict_obs
        next_observations = dict_next_obs
    return dict(
        observations=observations,
        actions=actions,
        rewards=np.array(rewards).reshape(-1, 1),
        next_observations=next_observations,
        terminals=np.array(terminals).reshape(-1, 1),
        agent_infos=agent_infos,
        env_infos=env_infos,
        goals=np.repeat(goal[None], path_length, 0),
        full_observations=dict_obs,
    )


def rollout(
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
