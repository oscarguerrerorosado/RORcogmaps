# HPC-VTA loop

## 🧠 Description

This repository contains a set of notebooks to train and analyze a hippocampal autoencoder following the HPC-VTA hypothesis: Novelty driven signals emerge in hippocampus and feeds back through VTA dopaminergic signals.

There are three folders included in the `.gitignore` file due to storage reasons.
 

### 🧩 Components

The repository includes several jupyter notebooks:

- **train_autoencoder.ipynb**   
  Trains the hippocampal autoencoder across the three versions of the T maze.

- **Ratemap_extraction.ipynb**  
  To accelerate data analysis, we first extract and save the embeddings of the model at every training epoch.

  Once saved, we can run the analysis notebooks.

- **Individual_exp_ratemap_analysis.ipynb**   
  Analyzes the performance progression of one sigle autoencoder. It is usefull to extract example ratempas and input reconstructions.

- **Group_exp_ratemap_analysis.ipynb**  
  Analyzes the joint performance progression of a set of autoencoder. It informs about means and stds metrics.


## 📬 Contact

For questions or issues, feel free to contact Oscar Guerrero Rosado (oscar.guerrerorosado@donders.ru.nl).

