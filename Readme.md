# Doing RL things!



# Logbook

### Lapland
I understood why the heck I am doing all this - for wizard level PhD degree at some point in my life. :)

### **09/10** 
I have added tensorboard summaries in different forms and flavour and got confused by what I saw: The first epoch, a lot of Episodes happened, which went only 8-11 steps, not until 1k steps. Reading in the paper and on [this one](https://towardsdatascience.com/deep-deterministic-policy-gradients-explained-2d94655a9b7b) I found that an update should happen every timestep though, with a couple of randomly sampled minibatches, not every 1000 steps. Also I understood the concept of the target network - it's pretty crazy, you mix in the target network at pretty unexpected points. Also the update mechanic of L = MSE( r(i) + Q'(i+1) - Q(i) ) is weird, I would have expected MSE( r(i) + Q(i+1) - Q(i) ). I have no clue why they did it like that, maybe other than experimenting until break of dawn :D gamma there is the discounting factor

Also I just understood why the process needs to be a Markov Process - because otherwise the replay buffer would not make sense! But because we view the problem as a Markov process, we can sample randomly from a replay buffer, without needing to look at sequential dependencies.

Another question to the article, it looks like the network the guy builds only contains 3 layers, one input, one hidden and one output layer for each net. Is that really all of it? I am just confused, as I thought stuff needs 30 something layers?

The policy gradient basic idea is nicely described [here](https://medium.com/@thechrisyoon/deriving-policy-gradients-and-implementing-reinforce-f887949bd63), though I found that most of the derivation didn't actually end up in the code, but was automatically done by pytorch. Also, policy gradients works on discrete actions and outputs probabilities for each action, which is why so many 'P's ended up in the formulas. It's generally advisable to stay clear of too many Ps.

Policy updating seems a bit weird - the formula has a lot of gradients and is notated weirdly - Q(s, a) instead of Q(s, u(s)). Also I don't really understand how to derive on the value function. The code however makes a lot more sense. The Q-function (critic) gives the expected returns for all the actions until infinity, so taking the negative of them means the lower the expected reward, the higher the loss, and that for me is somehow a good loss to backpropagate with. Pytorch does some magic with the optimizer to not also update the weights of the critic network. I guess if you ran both the critic and the actor optimizer, both nets would be updated.

Next Steps - Taking the pytorch code and using this for pendulum, then for qblade


### **09/11**
I started my day with trying to adapt the DDPG implementation of [this one](https://towardsdatascience.com/deep-deterministic-policy-gradients-explained-2d94655a9b7b) - there were some flaws in the repo such as: death conditions are not regarded at all. The main loop wasn't in the repo at all, no tensorboard reporting, etc

After a bit of tests, it works! It takes 25-ish episodes until convergence - it would be interesting to play around a bit with the hyperparameters...
![Pendulum](Screenshot-Pendulum.png)

I played around a bit with hyperparameters...

* changing noise: Seems to be important. With the correlated noise from the article it doesn't converge, with high (.4 and higher) np.uniform noise it does, after 25k steps. .5 after 15k, .6 after 12k but unstable, .7 too much
* changing batch size: not really much of a difference, worse with lower and a bit better with 64 as batchsize
* changing learning rate: higher - more interesting policy loss, but no convergence (until 35k)

As noise seemed that important, I am thinking about introducing parameter noise to the policy... I think that's a nice todo for tomorrow. Also getting the qbladeadapter running would be good :)

### **09/12**

I read a bit about TD3, I recognize the problem of the noisy Q-estimates (Q loss looks quite noisy in tensorboard). Also the algorithm doesn't seem much more complex than ddpg, I could try to implement it aswell.

QBlade keeps crashing a lot. Loading the library works, but already createInstance crashes sometimes. It seems to make a difference whether calling it via \_Z14createInstancev or via the link, but then not deterministically. Something seems badly wrong. It deterministically segfaults at setControlVars or initializeSimulation

WTF-Moment of the day: Suddenly ddpg converges reliably after 7-8 episodes without any noise at all, what I changed is nothing (except for adding parameter noise), but even after disabling it again it still converges quickly. WTF.


DDPG conclusion

* Added random exploration phase of 5k steps, super helpful.
* Changed the OUNoise from the paper to normal uniform noise, kinda helpful (maybe it had been enough to understand OUNoise and tweak it but meh)
* Added parameter noise to the policy, I think it's helpful, currently testing
* With enough noise it seems to produce quite robust results
* Sometimes, in the testing, though the policy has converged, it doesn't do sensful things. That could be due to it once making the hop up 'randomly' and then forgetting it
* I think it would be cool to experiment with TD3...

### **09/18**

I have spent the last couple days working on getting qblade running, but it's a quite iterative process of me trying something and qblade crashing, then me sending a mail to david and him fixing something.

Also somehow, DDPG has stopped converging with the very same hparams as before. I am back to playing around with it.

### **09/23**

I got qblade running and also have access to a small desktop computer where I can run my simulations much quicker. I have done several runs with the default scenario and ddpg, converging most of the time to setting all actions to 1. Q and policy loss approach 0 pretty quickly and for some reason all actions are the same (i.e. same outputs for all 5 control inputs). I have tried several runs, that was the same result every time - and the power output indeed raised. However, I have tried to maximize rotational speed, and it again converges with all actions to 1, not really delivering high rotational speeds. Same with me trying to optimize for high blade bending, the reward reduces over time as noise reduces, and the policy does not learn that shaking the blades could produce high plane bending. Even squaring the reward (in theory intensifying it) doesn't solve that, still converging to always 1 as output.


### **10/06**

Preparing the presentation for tomorrow, I tex-ed some formulas
Bellman
Q(s, a) &= r + \max_{a'}Q(s', a')
Q(s,a) = \mathbb{E}_{s' \sim \varepsilon}(r + \gamma \max_{a'}Q(s', a')|s,a)

Lookup-table
Q_{i+1}(s,a) \leftarrow (1-\alpha)Q_{i}(s,a) + \alpha(target)
target &= r + \gamma \max_{a'}Q_{i}(s', a')

DQN
L(\theta_i)=\mathbb{E}[(target -Q(s,a;\theta_i))^2]
target = r + \gamma \max_{a'} Q(s',a';\theta_{i-1})

DDPG
Q(s,a) = \mathbb{E}[r + \gamma Q(s', \mu(s))|s,a]

target = r + \gamma Q(s', \mu(s);\theta^Q_{i-1})
L(\theta^Q_i)=\mathbb{E}[(target -Q(s,a;\theta^Q_i))^2]
\Delta_{\theta^{\mu}} J(\theta^{\mu}) = \Delta_{\mu}Q(s, \mu(s))\Delta_{\theta^{\mu}}\mu(s)

### **10/07**

Todo-collection
* Setting a higher Gamma-value, as 0.99 is ca 0.37 after 100 iterations already
* Reworking the reward as rpm\*torque
* Trying way smaller fully connected networks
* Only having torque as control parameter, not blade pitch at all
* Having a look at the destruction problem (it should be maximizing at least a little)

Trying with smaller networks: critic [128, 64, 32] and actor [128, 32], increased gamma .9999 and reward as rpm\*torque

Actually trying various hyperparameters all yielded the same results, until I noticed that policy params aren't updating for some reason. Indifferently from learning rate, plain nothing happens. I compared the code with other implementations online (TODO: Add norms to the layers, should be better somehow), but their implementations were mostly equal.

### **10/08**

A talk with Matt hinted me at not having normalized anything. I will implement a normalization at the end of the random exploration phase to bring everything into the same order of magnitude.

So, after a while of trying, I noticed that I was actually dealing with the vanishing gradients problem. On the way there, I tried normalization without noticing that I actually made it far worse because I forgot the ^-1 to the normalizations and thus ended up making the big values way bigger and the small ones way smaller. Because I didn't notice, I tried fighting the vanishing gradients problem by reducing the complexity of the function approximators even further, ending up with both Q and policy as 1-layer network. I think I might want to increase that in the future...

Tries
* Subtract tip deflection from reward
* OU-Noise
* Reparameterize control so actions are gradients