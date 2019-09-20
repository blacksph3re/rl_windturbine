from .td3 import TD3

class Runner():
    """Carries out the environment steps and adds experiences to memory"""
    
    def __init__(self, hparams):
        self.hparams = hparams

        self.TD3 = TD3()
        self.env = env
        self.agent = agent
        self.replay_buffer = replay_buffer
        self.obs = env.reset()
        self.done = False
        
    def next_step(self, episode_timesteps, noise=0.1):
        
        action = self.agent.select_action(np.array(self.obs), noise=0.1)
        
        # Perform action
        new_obs, reward, done, _ = self.env.step(action) 
        done_bool = 0 if episode_timesteps + 1 == 200 else float(done)
    
        # Store data in replay buffer
        replay_buffer.add((self.obs, new_obs, action, reward, done_bool))
        
        self.obs = new_obs
        
        if done:
            self.obs = self.env.reset()
            done = False
            
            return reward, True
        
        return reward, done

    def evaluate_policy(policy, env, eval_episodes=100,render=False):
    """run several episodes using the best agent policy
        
        Args:
            policy (agent): agent to evaluate
            env (env): gym environment
            eval_episodes (int): how many test episodes to run
            render (bool): show training
        
        Returns:
            avg_reward (float): average reward over the number of evaluations
    
    """
    
    avg_reward = 0.
    for i in range(eval_episodes):
        obs = env.reset()
        done = False
        while not done:
            if render:
                env.render()
            action = policy.select_action(np.array(obs), noise=0)
            obs, reward, done, _ = env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes

    print("\n---------------------------------------")
    print("Evaluation over {:d} episodes: {:f}" .format(eval_episodes, avg_reward))
    print("---------------------------------------")
    return avg_reward

    def observe(env,replay_buffer, observation_steps):
    """run episodes while taking random actions and filling replay_buffer
    
        Args:
            env (env): gym environment
            replay_buffer(ReplayBuffer): buffer to store experience replay
            observation_steps (int): how many steps to observe for
    
    """
    
    time_steps = 0
    obs = env.reset()
    done = False

    while time_steps < observation_steps:
        action = env.action_space.sample()
        new_obs, reward, done, _ = env.step(action)

        replay_buffer.add((obs, new_obs, action, reward, done))

        obs = new_obs
        time_steps += 1

        if done:
            obs = env.reset()
            done = False

        print("\rPopulating Buffer {}/{}.".format(time_steps, observation_steps), end="")
        sys.stdout.flush()

def train(agent, test_env):
    """Train the agent for exploration steps
    
        Args:
            agent (Agent): agent to use
            env (environment): gym environment
            writer (SummaryWriter): tensorboard writer
            exploration (int): how many training steps to run
    
    """

    total_timesteps = 0
    timesteps_since_eval = 0
    episode_num = 0
    episode_reward = 0
    episode_timesteps = 0
    done = False 
    obs = env.reset()
    evaluations = []
    rewards = []
    best_avg = -2000
    
    writer = SummaryWriter(comment="-TD3_Baseline_HalfCheetah")
    
    while total_timesteps < EXPLORATION:
    
        if done: 

            if total_timesteps != 0: 
                rewards.append(episode_reward)
                avg_reward = np.mean(rewards[-100:])
                
                writer.add_scalar("avg_reward", avg_reward, total_timesteps)
                writer.add_scalar("reward_step", reward, total_timesteps)
                writer.add_scalar("episode_reward", episode_reward, total_timesteps)
                
                if best_avg < avg_reward:
                    best_avg = avg_reward
                    print("saving best model....\n")
                    agent.save("best_avg","saves")

                print("\rTotal T: {:d} Episode Num: {:d} Reward: {:f} Avg Reward: {:f}".format(
                    total_timesteps, episode_num, episode_reward, avg_reward), end="")
                sys.stdout.flush()


                if avg_reward >= REWARD_THRESH:
                    break

                agent.train(replay_buffer, episode_timesteps, BATCH_SIZE, GAMMA, TAU, NOISE, NOISE_CLIP, POLICY_FREQUENCY)

                # Evaluate episode
                if timesteps_since_eval >= EVAL_FREQUENCY:
                    timesteps_since_eval %= EVAL_FREQUENCY
                    eval_reward = evaluate_policy(agent, test_env)
                    evaluations.append(avg_reward)
                    writer.add_scalar("eval_reward", eval_reward, total_timesteps)

                    if best_avg < eval_reward:
                        best_avg = eval_reward
                        print("saving best model....\n")
                        agent.save("best_avg","saves")

                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1 

        reward, done = runner.next_step(episode_timesteps)
        episode_reward += reward

        episode_timesteps += 1
        total_timesteps += 1
        timesteps_since_eval += 1