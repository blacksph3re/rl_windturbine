# Doing RL things!



# Logbook

### Lapland
I understood why the heck I am doing all this - for wizard level PhD degree at some point in my life. :)

### **09/10** 
I have added tensorboard summaries in different forms and flavour and got confused by what I saw: The first epoch, a lot of Episodes happened, which went only 8-11 steps, not until 1k steps. Reading in the paper and on [this one](https://towardsdatascience.com/deep-deterministic-policy-gradients-explained-2d94655a9b7b) I found that an update should happen every timestep though, with a couple of randomly sampled minibatches, not every 1000 steps. Also I understood the concept of the target network - it's pretty crazy, you mix in the target network at pretty unexpected points. Also the update mechanic of L = MSE( r(i) + Q'(i+1) - Q(i) ) is weird, I would have expected MSE( r(i) + Q(i+1) - Q(i) ). I have no clue why they did it like that, maybe other than experimenting until break of dawn :D

Also I just understood why the process needs to be a Markov Process - because otherwise the replay buffer would not make sense! But because we view the problem as a Markov process, we can sample randomly from a replay buffer, without needing to look at sequential dependencies.