import copy
import ml_collections
from datetime import datetime

from Utils import *
from Training import *
from Model import *


def main(config: ml_collections.ConfigDict, log_df: pd.DataFrame):

    RNG = np.random.default_rng()    
    q = config.diagnostic.q
    k0 = config.diagnostic.k0
    thresh = config.diagnostic.thresh
    
    train_data, train_targets, test_data, test_targets, config.data.shape = download_data(config.data.name, "Data")
    
    # print(f"""Before preparing validation dataset...
    # The training data has a shape of {train_data.shape}
    # The test data has a shape of {test_data.shape}
    # The training targets has a shape of {train_targets.shape}
    # The test targets has a shape of {test_targets.shape}
    # The # of classes is {len(np.unique(train_targets))}""")
    
    config.data.num_classes = len(np.unique(train_targets))
    split_data = split_data_by_class(train_data, train_targets)
    split_valid_data = sample_data_per_class(config.data.num_validation, RNG, split_data)
    valid_data = np.concatenate([data for data in split_valid_data.values()])
    valid_targets = np.concatenate([np.full(len(data), clss, dtype=np.int8) for clss, data in split_valid_data.items()])
    
    worker_data_full, worker_targets_full = allocate_worker_datasets(config.worker.num, RNG, config.data.alloc_type, config.data.alloc_ratio, 
                                                           split_data, config.data.beta, config.data.shape)
    
    train_transforms = transforms.Compose(get_train_transforms(config.data.name))
    test_transforms = transforms.Compose(get_test_transforms(config.data.name))

    valid_data = torch.stack([test_transforms(valid_datum) for valid_datum in valid_data])
    valid_targets = torch.tensor(valid_targets)
    validset = torch.utils.data.TensorDataset(valid_data, valid_targets)
    validloader = torch.utils.data.DataLoader(validset, batch_size=config.data.batch_size, shuffle=False, num_workers=config.data.num_workers)

    test_data = torch.stack([test_transforms(test_datum) for test_datum in test_data])
    test_targets = torch.tensor(test_targets)
    testset = torch.utils.data.TensorDataset(test_data, test_targets)
    testloader = torch.utils.data.DataLoader(testset, batch_size=config.data.batch_size, shuffle=False, num_workers=config.data.num_workers)

    portions = config.worker.beta_list
    k = config.worker.k

    wandb.login()
    config.wandb.name = f'Run for {k_worker} workers and sample size of {portion*100}%'
    logged_config = flatten_dict(config.to_dict())
    run = wandb.init(project=config.wandb.project, job_type=config.wandb.job_type, name=config.wandb.name, config=logged_config)

    device = get_device()
    snet = get_model(config.train.model, config.data.num_classes)
    snet.to(device)
    criterion = config.train.loss_fn()
    criterion.to(device)
    cum_time = 0.0

    for k_worker in k:
        for portion in portions:
            print(f"Running Simulation for {k_worker} fastest workers and {portion*100}% Sample Size")
            worker_data = list([] for _ in range(config.worker.num))
            worker_targets = list([] for _ in range(config.worker.num))
            
            for worker in range(config.worker.num):
                samples_to_keep = int(len(worker_data_full[worker]) * portion)
                worker_data[worker] = worker_data_full[worker][:samples_to_keep]
                worker_targets[worker] = worker_targets_full[worker][:samples_to_keep]

            # plot_data_distribution(config.worker.num, worker_targets, config.wandb.name)
            worker_data = [torch.stack([train_transforms(datum) for datum in worker_datum]) for worker_datum in worker_data]
            worker_targets = [torch.tensor(targets) for targets in worker_targets]
            
            trainsets = [torch.utils.data.TensorDataset(worker_datum, worker_target) for worker_datum, worker_target in zip(worker_data, worker_targets)]
            
            # print(f"""After Preparing Validation...
            # The training data has a shape of {sum(map(len, worker_data))}
            # The training targets has a shape of {sum(map(len, worker_targets))}
            # The test data has a shape of {test_data.shape}
            # The test targets has a shape of {test_targets.shape}
            # The validation data has a total length of {len(valid_data)}
            # the validation targets has a total length of {len(valid_targets)}""")
            
            worker_epochs = generate_worker_epochs(config.worker.num, RNG, config.server.num_epochs, 
                                                config.worker.epoch.type, config.worker.epoch.is_random, 
                                                config.worker.epoch.mean, config.worker.epoch.std, 
                                                config.worker.epoch.beta, config.worker.epoch.coef)
            
            active_worker_matrix = generate_active_worker_matrix(config.worker.inact_prob, RNG, config.server.num_epochs, config.worker.num)     

            num_trainable_params = sum(p.numel() for p in snet.parameters())
            num_non_trainable_params = sum(b.numel() for b in snet.buffers())
            num_params = num_trainable_params + num_non_trainable_params

            # print(f"""The number of trainable parameters are {num_trainable_params}
            # The number of non-trainable parameters are {num_non_trainable_params}
            # The total number of parameters are {num_params}""")

            model_states = []
            
            for server_epoch in range(config.server.num_epochs):

                print(f"Server Epoch {server_epoch + 1} ...")
                epoch_start_time = datetime.now()
                
                agg_wgrads = {name:torch.zeros_like(param) for name, param in snet.named_parameters() if param.requires_grad}    
                agg_wbuffs = {name:torch.zeros_like(wbuff) for name, wbuff in snet.named_buffers()}
                
                worker_times = []
                num_active_worker = 0
                log_row = pd.Series()
                    
                for worker_idx, trainset in enumerate(trainsets):
                    
                    print(f"Training Worker {worker_idx + 1}...")
                    num_active_worker += 1
                    trainloader = torch.utils.data.DataLoader(trainset, batch_size=config.data.batch_size, shuffle=True, num_workers=config.data.num_workers, pin_memory=True)

                    worker_start_time = datetime.now()
                    
                    wnet = copy.deepcopy(snet)
                    woptimizer = config.worker.tx.fn(wnet.parameters(), lr=config.worker.tx.lr)
                    wgrads, wbuffs, log_row = train_client(wnet, trainloader, validloader, testloader,
                                                        device, criterion, woptimizer, 
                                                        worker_epochs[server_epoch, worker_idx],
                                                        worker_idx + 1, server_epoch + 1, log_row)
                    
                    worker_end_time = datetime.now()
                    worker_time = (worker_end_time - worker_start_time).total_seconds()
                    worker_times.append((worker_time, worker_idx, wgrads, wbuffs))
            
                worker_times.sort(key=lambda x: x[0])
                fastest_workers = worker_times[:k_worker]
                
                print(f"Aggregating gradients from the fastest {k_worker} workers...")
                for work_time, worker_idx, wgrads, wbuffs in fastest_workers:
                    agg_wgrads = {agg_name: agg_wgrad + wgrad for (agg_name, agg_wgrad), wgrad in zip(agg_wgrads.items(), wgrads.values())}
                    agg_wbuffs = {agg_name: agg_wbuff + wbuff for (agg_name, agg_wbuff), wbuff in zip(agg_wbuffs.items(), wbuffs.values())}
                    cum_time += work_time
                
                norm_agg_wgrads = {agg_name: agg_wgrad / k_worker for agg_name, agg_wgrad in agg_wgrads.items()}    
                norm_agg_wbuffs = {agg_name: agg_wbuff / k_worker for agg_name, agg_wbuff in agg_wbuffs.items()}
                
                print(f"Normalized Gradient for this Epoch == {global_norm(norm_agg_wgrads)}")

                soptimizer = config.server.tx.fn(snet.parameters(), lr=config.server.tx.lr)
                soptimizer.zero_grad()

                for name, param in snet.named_parameters():
                    if name in norm_agg_wgrads.keys():
                        if param.grad is None:
                            param.grad = norm_agg_wgrads[name]
                        else:
                            param.grad.data.copy_(norm_agg_wgrads[name])
                
                for name, buffer in snet.named_buffers():
                    if name in norm_agg_wbuffs.keys():
                        buffer.data.copy_(norm_agg_wbuffs[name])
                
                soptimizer.step()
                model_states.append(snet.state_dict())

                if server_epoch == k0 and k_worker == k[0]:
                    model0 = flatten_grads(model_states[0])

                svalid_loss, svalid_acc = eval_net(snet, validloader, device, criterion)
                log_row = log(log_row, server_epoch, {f'Server Validation Loss': np.mean(svalid_loss), f'Server Validation Accuracy': np.mean(svalid_acc)})
                
                stest_loss, stest_acc = eval_net(snet, testloader, device, criterion)
                log_row = log(log_row, server_epoch, {f'Server Test Loss': np.mean(stest_loss), f'Server Test Accuracy': np.mean(stest_acc)})

                if config.server.es.enable and server_epoch:       
                    if log_row['Server Test Accuracy'] - log_row['Server Test Accuracy'] < config.server.es_thresh:
                        config.server.es.wait += 1  
                    else:
                        config.server.es.wait = 0
                    if config.server.es.wait >= config.server.es.patience:
                        break

                epoch_end_time = datetime.now()
                print(f'Server Epoch {server_epoch + 1} is completed in {(epoch_end_time - epoch_start_time).total_seconds()} seconds.')
                log_row = log(log_row, server_epoch, {f'Server Run Time': (epoch_end_time - epoch_start_time).total_seconds()})
                
                wandb.log({
                    'Server Test Loss': np.mean(stest_loss),
                    'Server Test Accuracy': np.mean(stest_acc),
                    'Epoch': server_epoch + 1,
                    'Time': cum_time
                })

                if log_df.empty:
                    log_df = log_row.to_frame().T           
                else:
                    log_df.loc[len(log_df)] = log_row

                # log_df['Run'] = config.wandb.name
                # wandb.config.update(config.to_dict(), allow_val_change=True)
                # run.finish()
                                
                if server_epoch >= k0:
                    if server_epoch % 5 == 4:
                        print(f"Latest Model == {torch.norm(flatten_grads(model_states[server_epoch]))}\nOldest Grad == {model0}\nGrads from epoch {server_epoch//q} to be compared to == {torch.norm(flatten_grads(model_states[int(server_epoch//q)]))}")                
                        S = np.abs((np.log(torch.norm(model_states[server_epoch] - model0, p=2)**2) - np.log(np.linalg.norm(model_states[int(server_epoch//q)] - model0, p=2)**2)) / (np.log(server_epoch) - np.log(server_epoch/q)))
                        print(f"S ========= {S}")
                        if S < thresh:
                            config.server.num_epochs -= server_epoch 
                            break
       
        s = 1 / portions[0] * config.worker.num
        b_min = np.ceil((k_worker) * s / (k_worker + 1)) / s
        portions = [size for size in portions if size >= b_min]

    log_df['Run'] = config.wandb.name
    wandb.config.update(config.to_dict(), allow_val_change=True)
    run.finish()

    return snet, log_df
    
if __name__ == "__main__":
    configs = load_and_extract_configs('Params')    
    for idx, config in enumerate(configs):
        for run in range(int(config.train.num_runs)):
            curr_path = os.path.dirname(os.path.realpath(__file__))
            log_df = pd.DataFrame()
            federator_start_time = datetime.now()
            snet, log_df = main(config, log_df)
            federator_end_time = datetime.now()
            print(f'The federated learning process is completed in {(federator_end_time - federator_start_time).total_seconds()} seconds.')