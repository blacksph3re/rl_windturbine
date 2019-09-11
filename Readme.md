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
I started my day with trying to adapt the DDPG implementation of [this one](https://towardsdatascience.com/deep-deterministic-policy-gradients-explained-2d94655a9b7b) - there were some flaws in the repo such as: death conditions are not regarded at all. The main loop wasn't in the repo at all. Etc

After a bit of tests, it works! It takes 25-ish episodes until convergence - it would be interesting to play around a bit with the hyperparameters...
![Pendulum](Screenshot-Pendulum.png)
I played around a bit with hyperparameters, and it was pretty consistent over changing of noise and of batch size. Trying learning rate as next