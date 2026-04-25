EXPERIMENT 1: A Hippocampal autoencoder is trainned serially across three different evironments (datasets): T maze version 1, 2 and 3.

After the training phase where the model (.pth file) is saved every training epoch (100 epochs x 3 environments), the notebook Individual_exp_ratemap_analysis.ipynb extract the ratemaps of each model and save their reconstruction loss.

Models and ratemaps are stored in disk DATA(D:).

The notebook Group_exp_ratemap_analysis.ipynb goes through the extracted ratemaps to analyze how the model develops across training epochs and environments.

Results:
- Field sizes have a sharp increase at environment transitions signaling novelty. This increase in field size is positively correlated with the mode's reconstruction capability and negatively correlated with the model's cognitive map similarity.



Experiment 2:

Now, we want to train a model in a Double T maze.

First, we will train the model in a Double T maze that does not have any reward.

After being trained, we will present the trained model a dataset of the Double T maze where a reward object was introduced at the center of the maze.

We want to analyze if there is any signal of novelty in the model's embedding when observing that new object.



Experiment 3:

We will use novelty signals triggered by observing the new reward object to prioritize model training on those samples of the dataset.

We want to analyze if:
1. Does this lead to reward overrepresentation?
2. Does reward overrepresnetation represents an advantage for RL?
