# Pre-training Tiramisu Cost Model
This project uses pre-training technique to reduce the expensive data requirement of the cost model used in Tiramisu's autoscheduler. In this repository, one can choose to pre-train on only the access matrices (`/pretrain_access_matrices`), or pre-train on the entire computation vector (`/pretrain_comp_vector`). 

## Installation & Configuring the repository
Follow the same step as described in the Tiramisu Cost Model repo: https://github.com/Tiramisu-Compiler/cost_model. The step of training of the cost model after pre-training is also the same. 

## Generating Pre-training Dataset
Each datapoint in the pre-training dataset consists is a computation vector (whose composition is described in [Merouani and Boudaoud, et al.] (https://arxiv.org/abs/2403.11522). 

If user already have a large dataset used for training the cost model, they can directly run the file  `/pre_train/generate_comp_tensors_mp.py` (by changing the correct path to the dataset in the main function) to extract computation vectors from the programs and code transformations in the dataset. 

Alternatively, they can use the same data generator as used in generating random program for training Tiramisu auto scheduler's cost model (https://github.com/Tiramisu-Compiler/dataset-manipulation). One can generate as many programs (with respective transformations) as they want, and use the `/pre_train/generate_comp_tensors_mp.py` in this repo to parse them into computation vectors. 

## Training the auto-encoder
Run the file `/pre_train/train_comp_autoencoder.py` to train the auto-encoder. Remember to change the path to the dataset and the name of the weights to be saved in the file. 

## Training the cost model with pre-trained encoder
After an auto-encoder is pre-trained, navigate to the file `/conf/config.yml` and change the following field under **training**:

- **pretrained_weights_path**: change it to the corresponding path of your pre-trained weights
- **fine_tune_epoch**: This specify after which epoch you want to unfreeze the pre-trained weights. Before this epoch, the weights of the pre-trained weights are freezes. 
- **fine_tune_lr**: The learning rate applied to the pre-trained encoder after the weights are unfreeze. This is separate from the learning rate of the rest of the cost model, and it is recommended that **fine_tune_lr** is set less than **lr**. 

The documentation of other config option please refer to the original cost model repo: https://github.com/Tiramisu-Compiler/cost_model.

## Extra Notes on Pretraining on Comp Vector
Note that although we did not change the cost model architecture, there is a slight modifcation on the dataset for training the cost model: the upper bound of the loop in a computation vector is taken its $\text{log}_{10}$ value. This is because for some loop optimization, iteration bound are multiplied together. This can lead to very large value in the loop bound (~1e8). As the reconstruction loss of the autoencoder is MSE, this can easily lead to the loss to explode. **When training the cost model on existing dataset, remember to apply this to the input program, Or just use the `generate_dataset.py` file in this repository to regenerate dataset from the program annotation.** 