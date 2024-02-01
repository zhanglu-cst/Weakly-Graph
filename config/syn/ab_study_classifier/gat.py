import os

logger = dict(
        enable_wandb = True,
        entity = 'weakly_graph',
        project = 'syn_version5',
        name = 'GAT',
)

number_classes = 3

work_dir = os.path.join('./work_dir/', logger['project'], logger['name'])

total_number_itr = 10

root_dir = './data/syn_version5'
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
        number_subgraph_each_class = 1,
        init_subgraph_score = 1,
        number_classes = number_classes
)

classifier = dict(
        type = 'GAT_Predicter',
        in_feats = 10,
        output_dim = number_classes,
        hidden_feats = [512, 512, 512],
        emb_dim = 512,
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
                max_itr = 10000,  # change to itr to ssl
        ),
        cfg_pretrain_optimizer = dict(
                lr = 0.001
        ),
        cfg_dm = dict(
                number_itr = 1000,
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
                total_epoch = 10000,
                eval_interval = 1000,
                dataset = dataset,
                dataloader = dict(
                        type = 'build_simple_dataloader_DGL',
                        batch_size = 256,
                        number_workers = 0,
                ),
                eval_on = ['train', 'val', 'test'],
                max_ST_itr = 1,
                work_dir = work_dir,
                classifier = classifier,
        )
)

# trainer_classifier = dict(
#
# )

extract_key_subgraph = dict(
        type = 'Extract_Syn_Multi_Process',
        keep_subgraph_each_class = 10,
        number_classes = number_classes,
        min_node_number = 4,
        max_node_number = 16,
)
