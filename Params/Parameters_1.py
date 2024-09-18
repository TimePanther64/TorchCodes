import torch
from ml_collections import config_dict
import Model

#* Configuration
config = config_dict.ConfigDict()
# config.seed = 1 #! Not used currently
# config.experiment = 0 #! Not used currently

#* Training Configuration
config.train = config_dict.ConfigDict()
config.train.num_runs = 1
config.train.model = Model.lenet5
config.train.loss_fn = torch.nn.NLLLoss # torch.nn.CrossEntropyLoss

#* Data Configuration
config.data = config_dict.ConfigDict()
config.data.name = 'mnist'
config.data.alloc_type = 'uniform'
config.data.alloc_ratio = 2
config.data.beta = 0.5
config.data.batch_size = 128
config.data.num_validation = 100
config.data.shape = () #! TBD in Code 
config.data.num_classes = 0 #! TBD in Code 
config.data.num_workers = 0 if config.data.name == "mnist" else 4 #? It seems 0 works much better for Mnist

config.beta = config_dict.ConfigDict()
config.beta.mode = 'value' 
config.beta.value = 0.5
config.beta.index = 0

#*Server Configuration 
config.server = config_dict.ConfigDict()
config.server.num_epochs = 20
config.server.update_thresh = 0.75
config.server.es = config_dict.ConfigDict()
config.server.es.enable = False
config.server.es.threshold = 0.1 #! Not used currently
config.server.es.wait = 0
config.server.es.patience = 3

config.server.tx = config_dict.ConfigDict()
config.server.tx.fn = torch.optim.SGD
config.server.tx.lr = 0.1

#*Worker Configuration
config.worker = config_dict.ConfigDict()
config.worker.num = 50
config.worker.inact_prob = 0
config.worker.k = [5, 10, 15, 20]
config.worker.beta_list = [0.1, 0.5, 1.0]

config.worker.epoch = config_dict.ConfigDict()
config.worker.epoch.type = 'constant'
config.worker.epoch.mean = 3
config.worker.epoch.std = 2
config.worker.epoch.beta = 0.5
config.worker.epoch.coef = 20
config.worker.epoch.is_random = False

config.worker.tx = config_dict.ConfigDict()
config.worker.tx.fn = torch.optim.SGD
config.worker.tx.lr = 0.1
config.worker.tx.moment = 0
config.worker.tx.lr_decay_per_server_epoch = 1.0
config.worker.tx.type_decay_per_server_epoch = 'Geometric'
config.worker.tx.lr_decay_per_worker_epoch = 1.0
config.worker.tx.type_decay_per_worker_epoch = 'Geometric'

config.diagnostic = config_dict.ConfigDict()
config.diagnostic.q = 1.5
config.diagnostic.k0 = 5
config.diagnostic.thresh = 0.1
config.diagnostic.step = 5

#* Compressor Configuration
config.compressor = config_dict.ConfigDict()
config.compressor.enable = False

config.compressor.s2w = config_dict.ConfigDict()
config.compressor.s2w.enable = False
# config.compressor.s2w.comp = Compress.PermKCompressor
config.compressor.s2w.helper_comp = None
config.compressor.s2w.error_fb = None
config.compressor.s2w.clipping = [-8, 20] #! if helper_comp is defined this should be as well
config.compressor.s2w.rescale = False
config.compressor.s2w.k = None #! TBD in Code
config.compressor.s2w.prob = 0.0 #! TBD in Code
config.compressor.s2w.moment = config_dict.ConfigDict()
config.compressor.s2w.moment.enable = True 
config.compressor.s2w.moment.beta = config.worker.num ** (-2/3) #* One can unset enable_moment momentum by assigning 0 to beta


config.compressor.w2s = config_dict.ConfigDict()
config.compressor.w2s.enable = False
# config.compressor.w2s.comp = Compress.RandKCompressor
config.compressor.w2s.helper_comp = None
config.compressor.w2s.error_fb = None
config.compressor.w2s.clipping = [-8, 20] #! if helper_comp is defined this should be as well
config.compressor.w2s.rescale = False
config.compressor.w2s.k = None #! TBD in Code
config.compressor.w2s.prob = 0.0 #! TBD in Code
config.compressor.w2s.send_grad_diff = True

# #*Distilllation Configuration
# config.distillation = config_dict.ConfigDict()
# config.distillation.num_epochs = 5

# config.distillation.syn = config_dict.ConfigDict()
# config.distillation.syn.lr = config_dict.ConfigDict()
# config.distillation.syn.lr.minval = 0.01
# config.distillation.syn.lr.maxval = 0.1

# config.distillation.syn.data = config_dict.ConfigDict()
# config.distillation.syn.data.num = 50
# config.distillation.syn.data.mean = 5
# config.distillation.syn.data.std = 1
# config.distillation.syn.data.label_dominance = 100
# config.distillation.syn.data.label_as_prob = False
# config.distillation.syn.data.batch_size = 5

# config.distillation.tx = config_dict.ConfigDict()
# config.distillation.tx.name = optax.sgd.__name__ #! Must find another way to make this work 
# config.distillation.tx.lr = 0.02

config.wandb = config_dict.ConfigDict()
config.wandb.name = '' #! TBD in Code
config.wandb.project = 'Adaptive Strategy'
config.wandb.job_type = 'Adaptive Strategy'

# flat_config = {
#     'SEED': config.seed,
#     'EXPERIMENT': config.experiment,
    
#     'DATA_NAME': config.data.name,
#     'DATA_ALLOC_TYPE': config.data.alloc_type,
#     'DATA_ALLOC_RATIO': config.data.alloc_ratio,
#     'DATA_BETA': config.data.beta,
#     'DATA_BATCH_SIZE': config.data.batch_size,
#     'DATA_NUM_VALIDATION': config.data.num_validation,
    
#     'BETA_MODE': config.beta.mode,
#     'BETA_VALUE': config.beta.value,
#     'BETA_INDEX': config.beta.index,
    
#     'SERVER_NUM_EPOCHS': config.server.num_epochs,
#     'SERVER_UPDATE_THRESH': config.server.update_thresh,
#     'SERVER_ES': config.server.es,
#     'SERVER_ES_THRESH': config.server.es_thresh,
#     'SERVER_TX_FN': config.server.tx.fn.__name__,
#     'SERVER_TX_LR': config.server.tx.lr,
    
#     'WORKER_NUM': config.worker.num,
#     'WORKER_INACT_PROB': config.worker.inact_prob,
#     'WORKER_LOSS_FN': config.train.loss_fn.__name__,
#     'WORKER_BATCHWISE_UPDATE': config.train.batchwise_update,
#     'WORKER_METRIC_LOSS': config.train.metric.loss.__name__,
#     'WORKER_METRIC_ACC': config.train.metric.acc.__name__,
#     'WORKER_EPOCH_TYPE': config.worker.epoch.type,
#     'WORKER_EPOCH_MEAN': config.worker.epoch.mean,
#     'WORKER_EPOCH_STD': config.worker.epoch.std,
#     'WORKER_EPOCH_BETA': config.worker.epoch.beta,
#     'WORKER_EPOCH_COEF': config.worker.epoch.coef,
#     'WORKER_EPOCH_IS_RANDOM': config.worker.epoch.is_random,
#     'WORKER_TX_FN': config.worker.tx.fn.__name__,
#     'WORKER_TX_LR': config.worker.tx.lr,
#     'WORKER_TX_LR_DECAY_PER_SERVER_EPOCH': config.worker.tx.lr_decay_per_server_epoch,
#     'WORKER_TX_TYPE_DECAY_PER_SERVER_EPOCH': config.worker.tx.type_decay_per_server_epoch,
#     'WORKER_TX_LR_DECAY_PER_WORKER_EPOCH': config.worker.tx.lr_decay_per_worker_epoch,
#     'WORKER_TX_TYPE_DECAY_PER_WORKER_EPOCH': config.worker.tx.type_decay_per_worker_epoch,
    
#     'COMPRESSOR_ENABLE': config.compressor.enable,
#     'COMPRESSOR_S2W_ENABLE': config.compressor.s2w.enable,
#     'COMPRESSOR_S2W_ERROR_FB': config.compressor.s2w.error_fb,
#     'COMPRESSOR_S2W_CLIPPING': config.compressor.s2w.clipping,
#     'COMPRESSOR_S2W_RESCALE': config.compressor.s2w.rescale,
#     'COMPRESSOR_S2W_COMP': config.compressor.s2w.comp.__name__,
#     'COMPRESSOR_S2W_K': config.compressor.s2w.k,
#     'COMPRESSOR_S2W_PROB': config.compressor.s2w.prob,
#     'COMPRESSOR_W2S_ENABLE': config.compressor.w2s.enable,
#     'COMPRESSOR_W2S_ERROR_FB': config.compressor.w2s.error_fb,
#     'COMPRESSOR_W2S_CLIPPING': config.compressor.w2s.clipping,
#     'COMPRESSOR_W2S_RESCALE': config.compressor.w2s.rescale,
#     'COMPRESSOR_W2S_COMP': config.compressor.w2s.comp.__name__,
#     'COMPRESSOR_W2S_K': config.compressor.w2s.k,
#     'COMPRESSOR_W2S_PROB': config.compressor.w2s.prob,
#     'COMPRESSOR_W2S_BETA': config.compressor.w2s.beta
# }
# wandb_config = {
#     'FEDERATED_LEARNING_PARAMETERS': {
#         'RANDOM_SEED': config.seed,
#         'EXPERIMENT': config.experiment,
        
#         'TRAINING_DATASET': config.data.name,
#         'ALLOCATION_TYPE': config.data.alloc_type,
#         'ALLOCATION_RATIO': config.data.alloc_ratio,
#         'BATCH_LENGTH': config.data.batch_size,
#         'DATASET_BETA': config.data.beta,
#         'VALIDATION_SAMPLE_NUMBER_PER_CLASS': config.data.num_validation,
        
#         'BETA_MODE': config.beta.mode,
#         'BETA_INDEX': config.beta.index,
#         'BETA': config.beta.value,
        
#         'SERVER_MAX_ITERATION': config.server.num_epochs,
#         'CLIENT_UPDATE_THRESHOLD_RATIO': config.server.update_thresh,
#         'EARLY_STOPPING': config.server.es,
#         'EARLY_STOPPING_RATE': config.server.es_thresh,
#         'SERVER_LEARNING_RATE': config.server.tx.lr,
#         'SERVER_OPT_FUNCTION': config.server.tx.fn.__name__,
        
#         'CLIENT_NUMBER': config.worker.num,
#         'INACTIVE_PROBABILITY': config.worker.inact_prob,
#         'UPDATE_MODEL_EVERY_BATCH': config.train.batchwise_update,
#         'VARIED_LOCAL_ITERATION': config.worker.epoch.type,
#         'LOCAL_ITERATION_TYPE': config.worker.epoch.type,
#         'LOCAL_ITERATION_MEAN': config.worker.epoch.mean,
#         'LOCAL_ITERATION_STD': config.worker.epoch.std,
#         'LOCAL_ITERATION_BETA': config.worker.epoch.beta,
#         'LOCAL_ITERATION_COEFFICIENT': config.worker.epoch.coef,
#         'LOCAL_LEARNING_RATE': config.worker.tx.lr,
#         'LOCAL_LEARNING_RATE_DECAY_PER_SERVER_ITERATION': config.worker.tx.lr_decay_per_server_epoch,
#         'LOCAL_LEARNING_RATE_DECAY_PER_LOCAL_ITERATION': config.worker.tx.lr_decay_per_worker_epoch,
        
#         'COMPRESSOR_ENABLE': config.compressor.enable,
#         'COMPRESSOR_DL_ENABLE': config.compressor.s2w.enable,
#         'COMPRESSOR_DL_COMP': config.compressor.s2w.comp.__name__ if config.compressor.s2w.comp else 'None',
#         'COMPRESSOR_DL_ERROR_FEEDBACK': config.compressor.s2w.error_fb,
#         'COMPRESSOR_DL_CLIPPING': config.compressor.s2w.clipping,
#         'COMPRESSOR_DL_RESCLAE': config.compressor.s2w.rescale,
#         'COMPRESSOR_DL_K': config.compressor.s2w.k,
#         'COMPRESSOR_DL_BETA': 'None',
#         'COMPRESSOR_DL_PROB': config.compressor.s2w.prob,
#         'COMPRESSOR_UL_ENABLE': config.compressor.w2s.enable,
#         'COMPRESSOR_UL_COMP': config.compressor.w2s.comp.__name__ if config.compressor.w2s.comp else 'None',
#         'COMPRESSOR_UL_ERROR_FEEDBACK': config.compressor.w2s.error_fb,
#         'COMPRESSOR_UL_CLIPPING': config.compressor.w2s.clipping,
#         'COMPRESSOR_UL_RESCLAE': config.compressor.w2s.rescale,
#         'COMPRESSOR_UL_K': config.compressor.w2s.k,
#         'COMPRESSOR_UL_BETA': config.compressor.w2s.beta,
#         'COMPRESSOR_UL_PROB': config.compressor.w2s.prob,
        
        
#         # 'SYNTHETIC_DUMMY_RATE': config.distillation.syn.data.num,
#         # 'SYNTHETIC_DISTILLATION_RATE': config.distillation.syn.data.num,
#         # 'SYNTHETIC_DISTILLATION_EPOCH': config.distillation.num_epochs,
#         # 'SYNTHETIC_DATA_BATCH_LENGTH': config.distillation.syn.data.batch_size,
#         # 'SYNTHETIC_DATA_MEAN': config.distillation.syn.data.mean,
#         # 'SYNTHETIC_DATA_STD': config.distillation.syn.data.std,
#         # 'SYNTHETIC_DATA_NUMBER': config.distillation.syn.data.num,
#         # 'SYNTHETIC_LABEL_DOMINANCE': config.distillation.syn.data.label_dominance
#     }
# }
# if config.compressor.w2s.enable and not w2s_act_map[server_epoch]:
#             norm_agg_wgrads = {agg_name : agg_wgrad + prev_agg_wgrad for (agg_name, agg_wgrad), prev_agg_wgrad
#                            in zip(agg_wgrads.items(), prev_agg_wgrads.values())}