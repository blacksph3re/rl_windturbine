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