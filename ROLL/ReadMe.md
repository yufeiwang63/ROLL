### Code file structures/descriptions  
This folder includes all code files for training object-VAE, LSTM, the policy.
- launch_files/: the experiment launch file for each environment
- generate_LSTM_vae_only_dataset.py: generate or load pre-generated image datasets for pre-training the object-VAE and LSTM
- generate_vae_dataset.py: generate or load pre-generated image datasets for pre-training the scene-VAE
- LSTM_Model.py: the pytorch module of the LSTM we used in ROLL. The LSTM structure is: image-input -> conv layers -> flattened linear feature -> vae latent (also LSTM input) -> LSTM -> LSTM output as final object embedding. There is another decoder attached to the vae latent to train the VAE using image reconstruction loss.  See the code for more details.      
- LSTM_path_collector.py: a wrapper for collecting trajectories in the environment, should be used together with LSTM_wrapped_env.py  
- LSTM_schedule: different online training schedules for LSTM (but we finally used the same schedul for all tasks!)  
- LSTM_trainer.py: implement a trainer class for training the LSTM/object-VAE. Matching loss is also implemented in this file.  
- LSTM_wrapped_env.py: a wrapper on the underlying MujocoEnv that changes the observation, reward, and goal sampling. It needs a scene-VAE and a LSTM.  
    - observation: use the scene-VAE to encode the image from the underlying MujocoEnv to be latent vectors (see figure 1 in our paper).
    - reward: first segment the next image observation and goal image using the segmentation method, and then use LSTM to encode the segmented next image observation and the segmented goal image observation to object latent embeddings, compute the reward as the distance between these two latent embeddings (see figure 1 in our paper).  
    - goal sampling: randomly sample latent vector from the object-VAE latent, and then encode it using LSTM to get the final goal latent embedding (see figure 1 in our paper).  
    - it also implements various other functions. Please see the code for details.
- online_LSTM_replay_buffer: store the interaction trajectories with the environment. Hindsight Experience Replay is implemented when drawing smaples from the replay buffer.  
- onlineLSTMalgorithm: perform the ROLL algoirthm. In each epoch:
    - sample (s, a, s', g, r) tuples from the replay buffer (with HER) and use it with SAC to train the policy   
    - sample random image observations from the replay buffer to train the scene-VAE  
    - sample random trajectories from the replay buffer to train the LSTM  
- skewfit_full_experiments_LSTM.py: parsing all arguments from the launch file, pre-train (or load) scene-VAE/object-VAE/LSTM, pre-train openCV background subtraction module, prepare everything for the experiments and launch it.  
