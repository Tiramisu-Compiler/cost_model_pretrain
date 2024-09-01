import io
import hydra
import torch
from hydra.core.config_store import ConfigStore
from timeit import default_timer as timer
from utils.data_utils import *
from utils.modeling import *
from utils.train_utils import *
from pre_train.comp_autoencoder_fc_modeling import *

def define_and_load_model(conf):
    # Define the model
    autoencoder = AutoEncoder()

    #autoencoder.load_state_dict(torch.load("/scratch/cl5503/cost_model_auto_encoder/pre_train/saved_models/comps_w_expr_autoencoder_fc_linear_bottleneck.pt", map_location = train_device))
    autoencoder.eval()
    encoder = autoencoder.encoder.to(conf.testing.gpu)
    encoder.eval()
    model = Model_Recursive_LSTM_v2(
        input_size=conf.model.input_size,
        comp_embed_layer_sizes=list(conf.model.comp_embed_layer_sizes),
        drops=list(conf.model.drops),
        loops_tensor_size=8,
        device=conf.testing.gpu,
        pre_train_encoder=encoder
    )
    # Load the trained model weights
    model.load_state_dict(
        torch.load(
            conf.testing.testing_model_weights_path,
            map_location=conf.testing.gpu,
        )
    )
    model = model.to(conf.testing.gpu)
    
    # Set the model to evaluation mode
    model.eval()
    return model


def evaluate(conf, model):
    
    print("Loading the dataset...")
    val_ds, val_bl, val_indices, _ = load_pickled_repr(
        os.path.join(conf.experiment.base_path ,'pickled',conf.data_generation.dataset_name,'pickled_')+Path(conf.data_generation.valid_dataset_file).parts[-1][:-4], 
        max_batch_size = 1024, 
        store_device=conf.testing.gpu, 
        train_device=conf.testing.gpu
    )
    print("Evaluation...")
    start_time = timer()
    val_df = get_results_df(val_ds, val_bl, val_indices, model, train_device = conf.testing.gpu)
    end_time = timer()
    val_scores = get_scores(val_df)
    print("Evaluation time: ", end_time - start_time)
    return dict(
        zip(
            ["nDCG", "nDCG@5", "nDCG@1", "Spearman_ranking_correlation", "MAPE"],
            [item for item in val_scores.describe().iloc[1, 1:6].to_numpy()],
        )
    )


@hydra.main(config_path="conf", config_name="config")
def main(conf):
    print("Defining and loading the model using parameters from the config file")
    model = define_and_load_model(conf)
    print(f"Validating on the dataset: {conf.data_generation.valid_dataset_file}")
    scores = evaluate(conf, model)
    print("0.05 dataset")
    print(f"Evaluation scores are:\n{scores}")

if __name__ == "__main__":
    main()
