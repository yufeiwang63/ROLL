import numpy as np
from scripts.test_multiworld import show_obs

presampled_goals_path = 'data/local/goals/SawyerDoorHookResetFreeEnv-v1-goal.npy'
# presampled_goals_path = 'data/local/goals/SawyerPickupEnvYZEasy-v0-goal.npy'

presampled_goals = np.load(presampled_goals_path).item()
print(type(presampled_goals))
for key, val in presampled_goals.items():
    print(key)
    print(val.shape)

image_goals = presampled_goals['image_desired_goal']
print(image_goals.dtype)
for goal in image_goals:
    show_obs(goal)

  