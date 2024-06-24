import hydra
import gc
import shutil
import numpy
from timeit import default_timer as timer
from tqdm import tqdm
import sys
sys.path.insert(0, '..')
from hydra.core.config_store import ConfigStore
from utils.data_utils import *
from utils.modeling import *
from utils.train_utils import *

# For some loop optimization, iteration bound are time together
# This can lead to very large number in computation vector (~1e8)
# As the reconstruction loss of the autoencoder is MSE, this can easily lead to the loss to explode
# So we take log of the upper bound of the loop
# When using the cost model, this also apply to the input program
def thresholding_bound(x):
    sign = 1 if (x>0) else -1
    if abs(x) >= 10: # This is just an easier way to pick out upper bound, but is inefficient and may not be genrally applicable
        return (sign)*numpy.log10(abs(x))
    else:
        return x
def tensor_thresholding_bound(t):
    return t.apply_(thresholding_bound)

def generate_comp_vectors(input_q, output_q):
    comp_tensors_list = []
    loop_tensors_list = []

    process_id, programs_dict, repr_pkl_output_folder, train_device = input_q.get()
    function_name_list = list(programs_dict.keys())
    dropped_funcs = []
    for function_name in tqdm(function_name_list):
        nb_dropped = 0
    
        # Check whether this function should be dropped
        if drop_program(programs_dict[function_name], function_name):
            dropped_funcs.append(function_name)
            continue
        
        # Get the JSON representation of the program features
        program_json = programs_dict[function_name]["program_annotation"]
        
        # Extract the representation template for the datapoint
        try:
            (
                prog_tree,
                comps_repr_templates_list,
                loops_repr_templates_list,
                comps_placeholders_indices_dict,
                loops_placeholders_indices_dict
            ) = get_representation_template(
                programs_dict[function_name],
                train_device='cpu',
            )
        
        except (LoopsDepthException, NbAccessException):
            # If one of the two exceptions was raised, we drop all the schedules for that program and skip to the next program
            nb_dropped =+ len(
                programs_dict[function_name]["schedules_list"]
            )
            continue
        
        # For each schedule (sequence of transformations) collected for this function
        for schedule_index in range(len(programs_dict[function_name]['schedules_list'])):
            
            # Get the schedule JSON representation
            schedule_json = programs_dict[function_name]['schedules_list'][schedule_index]
            
            # Get the transformed execution time
            sched_exec_time = np.min(schedule_json['execution_times'])
            
            # Check if this schedule should be dropped
            if drop_schedule(programs_dict[function_name], schedule_index) or (not sched_exec_time):
                nb_dropped += 1
                continue
                
            
            # Fill the obtained template with the corresponsing schedule features
            try:
                comp_tensor, loop_tensor, comps_expr_repr  = get_schedule_representation(
                    program_json,
                    schedule_json,
                    comps_repr_templates_list,
                    loops_repr_templates_list,
                    comps_placeholders_indices_dict,
                    loops_placeholders_indices_dict,
                )
            except NbTranformationException:
                # If the number of transformations exceeded the specified max, we skip this schedule
                nb_dropped += 1 
                continue
            
            comps_expr_repr = comps_expr_repr.reshape((-1, MAX_EXPR_LEN*11))
            combined_tensor = torch.cat((comp_tensor.squeeze(), comps_expr_repr.squeeze()), dim = -1)
            
            # Add each part of the input to the local_function_dict to be sent to the parent process
            splitted_tensors = torch.split(combined_tensor.reshape((-1, 910 + 726)), 1, dim=0)
            
            comp_tensors_list.extend(splitted_tensors)
            
            #loops_tensor_list.append(loops_tensor)

    random.shuffle(comp_tensors_list)


    pkl_part_filename = repr_pkl_output_folder + '/pickled_representation_part_'+str(process_id)+'.pkl'
    with open(pkl_part_filename, 'wb') as f:
        pickle.dump(comp_tensors_list, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    # Send the file path to the parent process
    output_q.put((process_id, pkl_part_filename))

    #random.shuffle(loops_tensor_list)
    #print("Loops tensor: \n", loops_tensor_list[0])
    #print("Data length: ", len(comps_tensor_list))
    #np.save(comps_outfile, comps_tensor_list, allow_pickle = True)
    #np.save(loops_outfile, loops_tensor_list, allow_pickle = True)

class Dataset_parallel:
    def __init__(
        self,
        dataset_filename,
        max_batch_size = 64,
        drop_sched_func = None,
        drop_prog_func = None,
        speedups_clip_func = None,
        no_batching = False,
        store_device = "cpu",
        train_device = "cpu",
        repr_pkl_output_folder = "none",
        just_load_pickled_repr = False,
        nb_processes = 15,
        min_functions_per_tree_footprint=0
    ):

        # Structure to contain the batched inputs
        self.batched_X = []
        # Structure to contain the batched labels
        self.batched_Y = []
        # Number of all the dropped schedules
        self.nb_dropped = 0
        
        # Number of dropped schedules due to the drop schedule function only
        self.nb_pruned = 0
        # List of dropped functions 
        self.dropped_funcs = []

        # Number of loaded datapoints
        self.nb_datapoints = 0
        # number of batches that can fit in the GPU
        self.gpu_fitted_batches_index = -1
        
        self.nb_funcs_per_footprint = {}

        self.comp_tensors_datapoints_GPU = []
        self.comp_tensors_datapoints_CPU = []
        
        programs_dict = {}

        

        if just_load_pickled_repr: # Just load the existing repr
            print("Loading pkl files")
            storing_device = torch.device(0)
            for pkl_part_filename in tqdm(list(Path(repr_pkl_output_folder).iterdir())):
                pkl_part_filename = str(pkl_part_filename)
                if (torch.cuda.memory_allocated(storing_device.index) / torch.cuda.get_device_properties(storing_device.index).total_memory )> 0.70:
                    print("GPU almost full, switch to CPU")
                    storing_device = torch.device("cpu")
                with open(pkl_part_filename, 'rb') as f:
                    lst = pickle.load(f)
                #lst.apply_(thresholding_bound)
                lst = [tensor_thresholding_bound(tensor).to(storing_device) for tensor in lst]
                if storing_device == torch.device('cpu'):
                    self.comp_tensors_datapoints_CPU.extend(lst)
                else:
                    self.comp_tensors_datapoints_GPU.extend(lst)
            print("Finished loading pkl files")
            #self.comp_tensors_datapoints_GPU = torch.cat(self.comp_tensors_datapoints_GPU)
            #self.comp_tensors_datapoints_CPU = torch.cat(self.comp_tensors_datapoints_CPU)
            return
        else:
            # Separate the function according to the nb_processes parameter
            # Each process will extract the representation for a subset of functions and save that representation into a pkl file
            manager = multiprocessing.Manager()
            
            processs = []
            input_queue = manager.Queue()
            output_queue = manager.Queue()

            for i in range(nb_processes):
                processs.append(multiprocessing.Process(
                target=generate_comp_vectors, args=[input_queue, output_queue]))
            for process in processs:
                process.start()
                
            if dataset_filename.endswith("json"):
                with open(dataset_filename, "r") as f:
                    dataset_str = f.read()
                    
                programs_dict = json.loads(dataset_str)
                del dataset_str
                gc.collect()
            elif dataset_filename.endswith("pkl"):
                with open(dataset_filename, "rb") as f:
                    programs_dict = pickle.load(f)
                    
            functions_list = list(programs_dict.keys())
            random.Random(42).shuffle(functions_list)

            nb_funcs_per_process = (len(functions_list)//nb_processes)+1
            print("number of functions per process: ",nb_funcs_per_process)

            for i in range(nb_processes):
                process_programs_dict=dict(list(programs_dict.items())[i*nb_funcs_per_process:(i+1)*nb_funcs_per_process])
                input_queue.put((i, process_programs_dict, repr_pkl_output_folder, store_device))
                
            # This avoid the parents process finishing before child processes
            for i in range(nb_processes):
                process_id, pkl_part_filename = output_queue.get()
                # If we want to do batching immediatly after the processes are done, we read the pkl files from the child processes
                if not no_batching: 
                    with open(pkl_part_filename, 'rb') as f:
                        lst = pickle.load(f)
                    self.comp_tensors_datapoints.extend(lst)
            
            
        if no_batching:
            print("Parameter no_batching is True. Stopping after the PKL files were saved.")
            return
        del programs_dict
        #gc.collect()
        print( f"Number of datapoints {len(self.comp_tensors_datapoints)}")

    # Length of the dataset
    def __len__(self):
        return len(self.comp_tensors_datapoints)   

def load_data_into_pkls_parallel(datapath, nb_processes=15, repr_pkl_output_folder=None, overwrite_existing_pkl=False):
    
    if Path(repr_pkl_output_folder).is_dir() and overwrite_existing_pkl:
        shutil.rmtree(repr_pkl_output_folder)
        print('Deleted existing folder ', repr_pkl_output_folder)
        
    Path(repr_pkl_output_folder).mkdir(parents=True, exist_ok=False)
    print('Created folder ', repr_pkl_output_folder)
    
    # Read the JSONs and write the representation into the specified PKL path
    print("Loading data from: "+ datapath)
    dataset = Dataset_parallel(
        datapath,
        no_batching=True,
        just_load_pickled_repr=False,
        nb_processes=nb_processes, 
        repr_pkl_output_folder=repr_pkl_output_folder, 
        store_device="cpu", 
        train_device="cpu"
    )         
    return   

# Function to read the pkls written by the load_data_into_pkls_parallel function, batch the loaded data and return the batched data to be saved
def load_pickled_repr(repr_pkl_output_folder=None,max_batch_size = 1024, store_device="cpu", train_device="cpu", min_functions_per_tree_footprint=0):
    dataset = Dataset_parallel(
        None, 
        max_batch_size, 
        None, 
        repr_pkl_output_folder=repr_pkl_output_folder, 
        just_load_pickled_repr=True, 
        store_device=store_device, 
        train_device=train_device,
        min_functions_per_tree_footprint=min_functions_per_tree_footprint)

    return dataset

if __name__ == "__main__":
    base_path = '/scratch/cl5503/cost_model_auto_encoder/pre_train'
    datapath = "/scratch/cl5503/data/with_buffer_sizes/dataset_expr_batch550000-838143_val.pkl"
    dataset_name = "comp_and_expr_tensors_val"
    nb_processes = 64

    comp_tensor_pkl_output_folder = os.path.join(
        base_path, 
        'pickled',
        dataset_name,
        'pickled_'
    ) + Path(datapath).parts[-1][:-4]
    
    # If the pkl files haven't already been generated or if the directory is empty
    if not os.path.isdir(comp_tensor_pkl_output_folder) or not any(os.scandir(comp_tensor_pkl_output_folder)):
        print("Loading data, extracting comp tensors and writing it into pkl files")
        load_data_into_pkls_parallel(
            datapath,
            nb_processes = nb_processes,
            repr_pkl_output_folder = comp_tensor_pkl_output_folder,
            overwrite_existing_pkl = True
        )
    print(f"Reading the pkl files from {comp_tensor_pkl_output_folder} into memory for batching")
    parallel_dataset = load_pickled_repr(
        repr_pkl_output_folder = comp_tensor_pkl_output_folder,
        max_batch_size = 64,
        store_device = 'cpu',
        train_device = 'cpu'
    )

    
    # Shuffling batches to avoid having the same footprint in consecutive batches
    random.shuffle(parallel_dataset.comp_tensors_datapoints_GPU)
    random.shuffle(parallel_dataset.comp_tensors_datapoints_CPU)

    comp_tensors_dataset_path = os.path.join(
        base_path,
        "batched",
        dataset_name
    )
    # Write the first part of the batched validation data into a file
    comp_tensors_file_path_GPU = os.path.join(
        base_path, 
        "batched",
        dataset_name,   
        "comp_tensors_GPU.pt"
    )
    comp_tensors_file_path_CPU = os.path.join(
        base_path, 
        "batched",
        dataset_name,   
        "comp_tensors_CPU.pt"
    )
    
    if not os.path.exists(comp_tensors_dataset_path):
        os.makedirs(comp_tensors_dataset_path)

    print("Data length: ", len(parallel_dataset.comp_tensors_datapoints_GPU)+len(parallel_dataset.comp_tensors_datapoints_CPU))
        
    with open(comp_tensors_file_path_GPU, "wb") as comp_pickle_file:
        torch.save(parallel_dataset.comp_tensors_datapoints_GPU, comp_pickle_file)
    with open(comp_tensors_file_path_CPU, "wb") as comp_pickle_file:
        torch.save(parallel_dataset.comp_tensors_datapoints_CPU, comp_pickle_file)

    print("Finished. Comps and exprs from train")