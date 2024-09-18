import os, sys, importlib
import torch
import torchvision
from torchvision import transforms as transforms
import pandas as pd
import numpy as np
from numpy import typing as npt
from scipy.linalg import circulant
from matplotlib import pyplot as plt
import wandb
from datetime import datetime

def download_data(name_dataset:str, dir_dataset: str) -> tuple[npt.NDArray[np.int_], npt.NDArray[np.int_], npt.NDArray[np.int_], npt.NDArray[np.int_], npt.NDArray[np.int_]]:
    match name_dataset.casefold():
        case "mnist":
            dataset_downloader = torchvision.datasets.MNIST
        case "fmnist":
            dataset_downloader = torchvision.datasets.FashionMNIST
        case "cifar10":
            dataset_downloader = torchvision.datasets.CIFAR10
        case "cifar100": 
            dataset_downloader = torchvision.datasets.CIFAR100
        case _:
            raise ValueError(f"Failed to download {name_dataset}")
    trainset = dataset_downloader(dir_dataset, train = True, download=True)
    testset = dataset_downloader(dir_dataset, train = False, download=True)
    train_data, train_targets = np.array(trainset.data), np.array(trainset.targets)
    test_data, test_targets = np.array(testset.data), np.array(testset.targets)
    # print(f"The range of train data and test data are {np.ptp(train_data)}, {np.ptp(test_data)}.")
    # print(f"The data types of train data and test data are {train_data.dtype}, {test_data.dtype}")
    train_data = np.expand_dims(train_data, axis=-1) if train_data.ndim < 4 else train_data
    test_data = np.expand_dims(test_data, axis=-1) if test_data.ndim < 4 else test_data
    data_shape = train_data.shape[1:]
    return train_data, train_targets, test_data, test_targets, data_shape

#* Torch Preprocessing Functions
#*############################################################

def flatten_grads(grads:dict[str:torch.Tensor]):
    flat_grads = [grad.view(-1) for grad in grads.values()]
    return torch.cat(flat_grads)

def split_data_by_class_TORCH(dataset: torchvision.datasets.VisionDataset) -> dict[int, torch.Tensor]:
    split_datasets = {}
    for clss in dataset.class_to_idx.values():
        indices = torch.where(dataset.targets == clss)[0]
        split_datasets[clss] = dataset.data[indices]
    return split_datasets

def sample_data_per_class_TORCH(num_sample: int, split_data: dict[int, torch.Tensor]) -> dict[int, torch.Tensor]:
    sampled_split_data = {}
    for clss, val in split_data.items():
        indices = torch.randperm(len(val))[:num_sample]
        sampled_split_data[clss] = val[indices]
        mask = torch.ones(val.size(0), dtype=torch.bool)
        mask[indices] = False
        split_data[clss] = split_data[clss][mask]
    return sampled_split_data

def circulant_TORCH(tensor: torch.Tensor, dim: int) -> torch.Tensor:
    """Get a circulant version of the tensor along the {dim} dimension.
    The additional axis is appended as the last dimension.
    E.g. tensor=[0, 1, 2], dim=0 --> [[0, 1, 2],  [2, 0, 1],  [1, 2, 0]] """
    S = tensor.shape[dim]
    tmp = torch.cat([tensor.flip((dim,)), torch.narrow(tensor.flip((dim,)), dim=dim, start=0, length=S-1)], dim=dim)
    return tmp.unfold(dim, S, 1).flip((-1,))

def flatten_nested_lists_TORCH(data: list[torch.Tensor]) -> list:
    return [torch.cat(data[idx]) for idx in range(len(data))]

def allocate_worker_datasets_TORCH(num_worker: int, allocation_type: str, class_ratio: int,
                              split_data: dict[int, torch.Tensor], beta: float, data_shape: tuple[int, int, int]) -> tuple[list, list]:
    worker_data = list([] for _ in range(num_worker))
    worker_targets = list([] for _ in range(num_worker))
    num_dropped_samples = 0
    match allocation_type:
        case "class-workers":
            for worker in range(num_worker):
                if num_worker > len(split_data):
                    raise ValueError("Number of workers exceeds number of classes.")
                worker_data[worker] = split_data[worker]
                worker_targets[worker] = torch.full((len(split_data[worker]), ), worker, dtype=torch.int8)   
            # done with this case
        
        case "1/n-class-workers": # Each worker has data from n classes with one class being dominant, n = class_ratio
            # This doesn't work when the number of workers is higher than a certain number
            num_class_per_worker = num_worker * class_ratio
            if num_class_per_worker < len(split_data):
                raise ValueError("num_class_per_worker exceeds the number of available classes.")
            allocation_map = torch.cat([
                torch.tensor([num_class_per_worker // class_ratio + class_ratio - 1], dtype=torch.int16),
                torch.full((num_worker - 1, ), class_ratio - 1, dtype=torch.int16)
                ])   
            # torch.arange(num_worker + class_ratio - 1, num_class_per_worker, class_ratio - 1, dtype=torch.int8)
            allocation_order = circulant(torch.arange(num_worker, dtype=torch.int8), 0)
            allocation_row = 0
            # print(allocation_map, '\n', allocation_order)   
            for target, data in split_data.items():
                sample_number = len(data)
                class_length = sample_number // (num_class_per_worker)
                excess_samples = sample_number % num_class_per_worker
                num_dropped_samples += excess_samples
                class_data = data[:-excess_samples, :, :, :] if excess_samples else data[:]
                class_data = class_data.reshape((num_class_per_worker), class_length, *data_shape)
                split_class_data = torch.split(class_data, allocation_map.tolist(), dim=0)
        
                for worker in range(num_worker):
                    worker_idx = allocation_order[allocation_row, worker]
                    temp = split_class_data[worker].reshape(-1, *data_shape)
                    worker_data[worker_idx].append(temp)
                    # print(temp.shape)
                    worker_targets[worker_idx].append(torch.full((len(temp), ), target, dtype=torch.int8))
                allocation_row += 1
            # done with this case
        
        case "uniform":
            num_class_per_worker = num_worker
            for target, data in split_data.items():
                sample_number = len(data)
                class_length = sample_number // (num_class_per_worker)
                excess_samples = sample_number % num_class_per_worker
                num_dropped_samples += excess_samples
                class_data = data[:-excess_samples, :, :] if excess_samples else  data[:]
                class_data = class_data.reshape((num_class_per_worker), class_length, *data_shape)
                # print(class_data.shape)
                for worker in range(num_worker):
                    worker_data[worker].append(class_data[worker])
                    worker_targets[worker].append(torch.full((class_length, ), target, dtype=torch.int8))
            # done with this case
        
        case "random":
            for target, data in split_data.items():
                allocation_map = torch.sort(torch.randint(len(data), (num_worker - 1,)))[0]
                split_sizes = torch.diff(torch.cat((torch.tensor([0]), allocation_map, torch.tensor([len(data)]))))
                class_data = torch.split(data, split_sizes.tolist())

                for worker in range(num_worker):
                    worker_data[worker].append(class_data[worker])
                    worker_targets[worker].append(torch.full((len(class_data[worker]), ), target, dtype=torch.int8))
            # done with this case
        
        case "dirichlet":
            dirichlet_dist = torch.distributions.Dirichlet(torch.full((num_worker,), beta))
            sample_rate = dirichlet_dist.sample()
            allocation_map = torch.cumsum((sample_rate * len(data)).long(), dim=0)[:-1]
            class_data = torch.tensor_split(data, allocation_map.tolist())

            for worker in range(num_worker):
                worker_data[worker].append(class_data[worker])
                worker_targets[worker].append(torch.full((len(class_data[worker]),), target, dtype=torch.int8))
        case _:
            raise ValueError("Invalid allocation type")
        
    # print(f"The number of dropped samples is equal to {num_dropped_samples}")
    if allocation_type != "class-workers":
        worker_data, worker_targets = flatten_nested_lists(worker_data), flatten_nested_lists(worker_targets)
    return worker_data, worker_targets

def generate_worker_epochs_TORCH(num_worker: int, max_iteration: int, iteration_type: str, is_random: bool,
                           mean: float, std: float, beta: float, coefficient: int) -> torch.Tensor:
    match iteration_type:
        case 'constant':
            local_iteration_distribution = torch.full((num_worker,), mean, dtype=torch.int)
        
        case 'uniform':
            local_iteration_distribution = torch.empty(num_worker).uniform_(mean - std, mean + std).round().int()
        
        case 'gaussian':
            if is_random:
                distribution_mean = (torch.distributions.Dirichlet(torch.full((num_worker,), beta)).sample() * coefficient).round()
                local_iteration_distribution = torch.maximum(
                    torch.normal(mean=distribution_mean, std=std).round(), 
                    torch.tensor(1.0)).int()
                local_iteration_distribution = local_iteration_distribution.expand(max_iteration, num_worker)
            else:
                local_iteration_distribution = torch.maximum(
                    torch.normal(mean=mean, std=std, size=(num_worker,)).round(), 
                    torch.tensor(1.0)).int()
        
        case 'exponential':
            if is_random:
                distribution_mean = (torch.distributions.Dirichlet(torch.full((num_worker,), beta)).sample() * coefficient).round()
                local_iteration_distribution = torch.maximum(
                    torch.distributions.Exponential(distribution_mean).sample((max_iteration, num_worker)).round(), 
                    torch.tensor(1.0)).int()
            else:
                local_iteration_distribution = torch.maximum(
                    torch.distributions.Exponential(mean).sample((num_worker,)).round(), 
                    torch.tensor(1.0)).int()
        
        case 'dirichlet':
            local_iteration_distribution = (torch.distributions.Dirichlet(torch.full((num_worker,), beta)).sample() * coefficient).round().int()
        
        case _:
            raise ValueError("Invalid iteration type")
    
    print(f"The local iteration distribution is {local_iteration_distribution.shape} shaped")
    
    if local_iteration_distribution.ndim == 1:
        local_iteration_distribution = local_iteration_distribution.unsqueeze(0).expand(max_iteration, num_worker)
    
    print(f"The local iteration distribution is expanded to {local_iteration_distribution.shape}")
    return local_iteration_distribution

def generate_active_worker_matrix_TORCH(inactive_probability: float, max_iteration: int, num_worker: int) -> torch.Tensor:
    active_prob_matrix = torch.full((max_iteration, num_worker), 1 - inactive_probability, dtype=torch.float32)
    active_worker_matrix = torch.bernoulli(active_prob_matrix).bool()
    return active_worker_matrix

#* Numpy Preprocessing Functions
#*############################################################

def split_data_by_class(data: npt.NDArray[np.int_], labels: npt.NDArray[np.int_]) -> dict[int, npt.NDArray[np.int_]]:
    split_data = {}
    for clss in np.unique(labels):
        indices = np.where(labels == clss)[0]
        split_data[clss] = data[indices]
    return split_data

def sample_data_per_class(sample_number: int, RNG: np.random.Generator, split_data: dict[int, npt.NDArray[np.int_]]) -> dict[int, npt.NDArray[np.int_]]:
    sampled_split_data = {}
    for label, val in split_data.items():
        indices = RNG.choice(len(val), sample_number, replace=False)
        sampled_split_data[label] = val[indices]
        split_data[label] = np.delete(split_data[label], indices, axis=0)
    return sampled_split_data

def flatten_nested_lists(data: list[list]) -> list:
    return [np.concatenate(data[idx]) for idx in range(len(data))]

def allocate_worker_datasets(worker_number: int, RNG: np.random.Generator, allocation_type: str, class_ratio: int, 
                             split_data: dict[int, npt.NDArray[np.int_]], beta: float, data_shape: tuple[int, int, int]) -> tuple[list, list]:
    
    worker_data = list([] for _ in range(worker_number))
    worker_labels = list([] for _ in range(worker_number))
    total_dropped_samples = 0

    match allocation_type:
        case "class-workers": # Each worker has data from a single class
            for worker in range(worker_number):
                worker_data[worker] = split_data[worker]
                worker_labels[worker] = np.full(len(split_data[worker]), worker, dtype=np.int8)   
            # done with this case
        
        case "1/n-class-workers": # Each worker has data from n classes with one class being dominant, n = class_ratio
            # This doesn't work when the number of workers is higher than a certain number
            class_number_per_worker = worker_number * class_ratio
            allocation_map = np.arange(worker_number + class_ratio - 1, class_number_per_worker, class_ratio - 1, dtype=np.int8)
            allocation_order = circulant(np.arange(worker_number, dtype=np.int8)).T
            allocation_row = 0
            # print(allocation_map, '\n', allocation_order)   
            for label, data in split_data.items():
                sample_number = len(data)
                class_length = sample_number // (class_number_per_worker)
                excess_samples = sample_number % class_number_per_worker
                total_dropped_samples += excess_samples
                classed_data = data[:-excess_samples, :, :] if excess_samples else data[:]
                classed_data = classed_data.reshape((class_number_per_worker), class_length, *data_shape)
                split_classed_data = np.split(classed_data, allocation_map, axis=0)
        
                for worker in range(worker_number):
                    worker_idx = allocation_order[allocation_row, worker]
                    temp = split_classed_data[worker].reshape(-1, *data_shape)
                    worker_data[worker_idx].append(temp)
                    # print(temp.shape)
                    worker_labels[worker_idx].append(np.full(len(temp), label, dtype=np.int8))
                allocation_row += 1
            # done with this case
        
        case "uniform":
            class_number_per_worker = worker_number
            for label, data in split_data.items():
                sample_number = len(data)
                class_length = sample_number // (class_number_per_worker)
                excess_samples = sample_number % class_number_per_worker
                total_dropped_samples += excess_samples
                uniformly_classed_data = data[:-excess_samples, :, :] if excess_samples else data[:]
                # print(uniformly_classed_data.shape)
                uniformly_classed_data = uniformly_classed_data.reshape(class_number_per_worker, class_length, *data_shape)
                for worker in range(worker_number):
                    worker_data[worker].append(uniformly_classed_data[worker])
                    worker_labels[worker].append(np.full(class_length, label, dtype=np.int8))
            # done with this case
        
        case "random":
            for label, data in split_data.items():
                allocation_map = np.sort(RNG.integers(len(data), size = worker_number - 1))
                randomly_classed_data = np.split(data, allocation_map, axis=0)

                for worker in range(worker_number):
                    worker_data[worker].append(randomly_classed_data[worker])
                    worker_labels[worker].append(np.full(len(randomly_classed_data[worker]), label, dtype=np.int8))
            # done with this case
        
        case "dirichlet":
            sample_rate = RNG.dirichlet(np.full(worker_number, beta), size=len(split_data))
            for label, data in split_data.items():
                allocation_map = np.cumsum(np.int32(sample_rate[label] * len(data)))[:-1]
                dirichlet_classed_data = np.split(data, allocation_map, axis=0)
                
                for worker in range(worker_number):
                    worker_data[worker].append(dirichlet_classed_data[worker])
                    worker_labels[worker].append(np.full(len(dirichlet_classed_data[worker]), label, dtype=np.int8))
            #done with this case
        case _:
            raise ValueError("Invalid allocation type")
        
    print(f"The number of dropped samples is equal to {total_dropped_samples}")

    if allocation_type != "class-workers":
        worker_data, worker_labels = flatten_nested_lists(worker_data), flatten_nested_lists(worker_labels)
        
        for worker in range(worker_number):
            randomized_idx = np.arange(len(worker_data[worker]))
            RNG.shuffle(randomized_idx)
            worker_data[worker], worker_labels[worker] = worker_data[worker][randomized_idx], worker_labels[worker][randomized_idx]
                                
    return worker_data, worker_labels

def generate_worker_epochs(worker_number:int, RNG: np.random.Generator, max_iteration:int, iteration_type: str, does_it_vary:bool,  
                                      mean: float, std: float, beta:float, coefficient: int) -> np.ndarray:
    match iteration_type:
        case 'constant':
            local_iteration_distribution = np.full(worker_number, mean)
            #done with this case
        case 'uniform':
            local_iteration_distribution = RNG.uniform(mean - std, mean + std, size=worker_number)
            #done with this case
        case 'gaussian':
            if does_it_vary:
                distribution_mean = (RNG.dirichlet(np.full(worker_number, beta)) * coefficient).round()
                local_iteration_distribution = np.maximum(RNG.normal(distribution_mean, std, (max_iteration, worker_number)).round(), 1)
            else:
                local_iteration_distribution = np.maximum(RNG.normal(mean, std, worker_number).round(), 1)
            #done with this case
        case 'exponential':
            if does_it_vary:
                distribution_mean = (RNG.dirichlet(np.full(worker_number, beta)) * coefficient).round()
                local_iteration_distribution = np.maximum(RNG.exponential(distribution_mean, (max_iteration, worker_number)).round(), 1)
            else:
                local_iteration_distribution = np.maximum(RNG.exponential(mean, size=worker_number).round(), 1)
                #done with this case
        case 'dirichlet':
            local_iteration_distribution = (RNG.dirichlet(np.full(worker_number, beta)) * coefficient).round()
            #done with this case
        case _:
            raise ValueError("Invalid iteration type")
    # print(f"The local iteration distribution is {local_iteration_distribution.shape} shaped")
    if local_iteration_distribution.ndim == 1:
        local_iteration_distribution = np.repeat(local_iteration_distribution.reshape(-1, worker_number), max_iteration, axis=0)
    # print(f"The local iteration distribution is expanded to {local_iteration_distribution.shape}")
    return np.int16(local_iteration_distribution)

def generate_active_worker_matrix(inactive_probability: float, RNG: np.random.Generator, max_iteration: int, worker_number: int):
    active_worker_matrix = RNG.choice([False, True], (max_iteration, worker_number), True,
                                            [inactive_probability, 1-inactive_probability])
    return active_worker_matrix

#* Utility Functions
#*############################################################

def distance_based_diagnostic(theta_0: int, theta_n: int, theta_nq: int, n: int, q: int, k0: int, thresh: int):
    if n == q**(int(n // q) + 1) and (n // q) >= k0:
        S = (np.log(np.linalg.norm(theta_n - theta_0)**2) - np.log(np.linalg.norm(theta_nq - theta_0)**2)) / (np.log(n) - np.log(n/q))     
        return S < thresh
    else:
        return False

def get_train_transforms(dataset_type:str):
    match dataset_type.casefold():
        case "mnist" | "fmnist":
            train_transforms = [
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ]
        case "cifar10" | "cifar100":
            train_transforms = [
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation((-7,7)),
                transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize((0.491, 0.482, 0.446), (0.247, 0.243, 0.261))]
    return train_transforms

def get_test_transforms(dataset_type:str):
    match dataset_type.casefold():
        case "mnist" | "fmnist":
            test_transforms = [
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ]
        case "cifar10" | "cifar100":            
            test_transforms = [
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize((0.491, 0.482, 0.446), (0.247, 0.243, 0.261))]
    return test_transforms

def plot_data_distribution(worker_number: int, worker_labels: list[np.ndarray], run_name:str):
    
    if worker_number <= 20:
        plt.figure(1, figsize=(int(8 * worker_number / 10), 8))
    else:
        plt.figure(1, figsize=(int(4 * worker_number / 10), 16))
    
    for worker in range(len(worker_labels)):
        if worker_number <= 20:
            plt.subplot(5, int(np.ceil(worker_number / 5)), worker + 1)
        else:
            plt.subplot(10, int(np.ceil(worker_number / 10)), worker + 1)
        plt.hist(worker_labels[worker], color="lightblue", ec="red", align="left", bins=np.arange(11))
        plt.title("worker " + str(worker + 1))
    
    plt.suptitle(run_name)
    plt.tight_layout()
    plt.show()
    
    # logs_dir = os.path.join(os.path.dirname(__file__), 'Logs')
    # if not os.path.exists(logs_dir):
    #     os.makedirs(logs_dir)
    # plt.savefig(os.path.join(logs_dir, f"{run_name}_Data_Distribution.png"))
    # plt.close()

def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key.upper(), sep=sep).items())
        else:
            items.append((new_key.upper(), v))
    return dict(items)

def load_and_extract_configs(dir_path, include=None, exclude=None):
    """
    Load Python modules dynamically from a specified directory and extract a 'config' variable from each,
    with options to include specific modules or exclude specified modules.

    Parameters:
    - dir_path (str): The local path to the directory containing Python modules.
    - include (list): Optional list of module filenames to specifically include (without '.py' extension).
    - exclude (list): Optional list of module filenames to exclude (without '.py' extension).

    Returns:
    - list: A list of all the 'config' variables extracted from the modules.
    """
    if exclude is None:
        exclude = []

    curr_path = os.path.dirname(os.path.realpath(__file__))
    param_files = os.listdir(os.path.join(curr_path, dir_path))

    if include is not None:
        param_files = [f.split('.')[0] for f in param_files if f.split('.')[0] in include]
    else:
        param_files = [f.split('.')[0] for f in param_files if (f.endswith('.py') and f.split('.')[0] not in exclude)]
    configs = []

    if param_files:
        for f in param_files:
            module_name = f"{dir_path.replace('/', '.')}.{f}"
            try:
                module = importlib.import_module(module_name)
                print(f"Loading Configs from {f}.py...")
                config = getattr(module, 'config')
                configs.append(config)
                print("Done.")
            except AttributeError as e:
                print(f"Could not find 'config' in {f}.py. Skipping.")
                print(f"The error is {e}")
            except ImportError as e:
                print(f"Failed to import {f}.py: {str(e)}")
            finally:
                if module_name in sys.modules:
                    del sys.modules[module_name]
    else:
        print("No Python files found based on the include/exclude criteria.")

    return configs

def log(row:pd.Series, step:int, logees:dict[str, any]):
    for key, data in logees.items():
        if isinstance(data, list):
            for idx, datum in enumerate(data):
                wandb.log(data = {key: np.mean(datum)})
        else:
            wandb.log(data = {key: data})# , step = step)
    return pd.Series(logees) if row.empty \
    else pd.concat([row, pd.Series(logees)], axis=0)

def get_run_name(config, run):
    name = ""
    name += f"Torch|"
    
    #* Model
    name += f"{config.train.model.__name__.lower()}|"

    #* Data
    name += f"{config.data.name}|"
    name += f"BS-{config.data.batch_size}|"
    
    #* Server
    name += f"SNE-{config.server.num_epochs}|"
    name += f"STX-{config.server.tx.fn.__name__}|"
    name += f"SLR-{config.server.tx.lr}|"
    name += f"SLRDR-{config.worker.tx.lr_decay_per_server_epoch}|"
    
    #* Worker
    name += f"WNE-{config.worker.epoch.mean}|"
    name += f"WN-{config.worker.num}|"
    name += f"WTX-{config.worker.tx.fn.__name__}|"
    name += f"WLR-{config.worker.tx.lr}|"
    name += f"WLRDR-{config.worker.tx.lr_decay_per_worker_epoch}|"
    
    #* Compressor
    if config.compressor.enable:
        name += f"C-{config.compressor.enable}|"
        name += f"S2W-{config.compressor.s2w.enable}|"
        name += f"S2W-{config.compressor.s2w.comp.__name__}|"
        name += f"W2S-{config.compressor.w2s.enable}|"
        name += f"W2S-{config.compressor.w2s.comp.__name__}|"
    
    #* Run
    name += f"R#{run}|"
    name += datetime.now().strftime("%y%m%d%H%M%S")
    return name

def global_norm(x:dict[str:torch.Tensor])->torch.Tensor:
    return torch.norm(torch.tensor([torch.norm(val) for val in x.values()])).item() if len(x) > 0 else torch.tensor(-1).item()

def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device