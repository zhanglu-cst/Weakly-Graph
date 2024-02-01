import os

logger = dict(
        enable_wandb = True,
        entity = 'weakly_graph',
        project = 'IMDB_BINARY',
        name = 'baseline',
)

resume = True
number_classes = 2

work_dir = os.path.join('./work_dir/', logger['project'], logger['name'])

total_number_itr = 10

root_dir = './data/IMDB-BINARY'
dataset = dict(
        train = dict(
                type = 'SYN_Dataset',
                root_dir = root_dir,
                split = 'train',
        ),
        val = dict(
                type = 'SYN_Dataset',
                root_dir = root_dir,
                split = 'val',
        ),
        test = dict(
                type = 'SYN_Dataset',
                root_dir = root_dir,
                split = 'test',
        )
)

key_subgraph = dict(
        type = 'Key_SubGraph_Syn',
        init_subgraph_filename = 'key_subgraph_each_class.pkl',
        root_dir = root_dir,
        init_subgraph_score = 100,
        number_classes = number_classes
)

classifier = dict(
        type = 'GIN',
        num_node_emb_list = [10],
        num_edge_emb_list = [10],
        num_layers = 3,
        emb_dim = 512,
        JK = 'last',
        output_dim = number_classes,
)

trainer = dict(
        type = 'Trainer_KeySubgraph_Syn',
        number_classes = number_classes,
        number_of_process = 40,
        cfg_model_pretrain = dict(
                type = 'Pretrain_Wrapper_Model',
                cfg_classifier = classifier,
                number_classes = number_classes,
                lamda_graph = 1,
        ),
        cfg_pretrain_loader = dict(
                num_worker = 0,
                batch_size = 64,
                max_itr = 0,  # change to itr to ssl
        ),
        cfg_pretrain_optimizer = dict(
                lr = 0.001
        ),
        cfg_dm = dict(
                number_itr = 0,
                lamda1 = 0.0001,
        ),
        pseudo_label_key = 'all_pseudo_labels_mul',
        cfg_finetune_loader = dict(
                batch_size = 64,
                epoch = 100,
        ),
        cfg_finetune_optimizer = dict(
                lr = 0.0001,
        ),
        cfg_trainer_finetune = dict(
                optimizer_classifier = dict(
                        name = 'AdamW',
                        lr = 0.001,
                ),
                train_val_dataset = dict(
                        type = 'DGL_Dataset',
                        max_upsample = 1,
                ),
                stop_itr = [500000, ],
                total_epoch = 5000,
                eval_interval = 20,
                dataset = dataset,
                dataloader = dict(
                        type = 'build_simple_dataloader_DGL',
                        batch_size = 256,
                        number_workers = 0,
                ),
                eval_on = ['train', 'val', 'test'],
                number_classes = number_classes,
                max_ST_itr = 1,
                work_dir = work_dir,
                classifier = classifier,
        )
)

extract_key_subgraph = dict(
        type = 'Extract_Syn_Multi_Process',
        keep_subgraph_each_class = 20,
        number_classes = number_classes,
        min_node_number = 4,
        max_node_number = 25,
)
