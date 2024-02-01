logger = dict(
        enable_wandb = True,
        entity = 'weakly_graph',
        project = 'syn',
        name = 'run_good',
)
number_classes = 3

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

# pseudo_label = dict(
#         type = 'Vote_Syn',
#         number_classes = number_classes,
# )

pseudo_label = dict(
        type = 'Denoiser_Syn',
        number_classes = number_classes,
        number_of_process = 30,
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

trainer_classifier = dict(
        optimizer_classifier = dict(
                name = 'AdamW',
                lr = 0.001,
        ),
        train_val_dataset = dict(
                type = 'DGL_Dataset',
                max_upsample = 1,
        ),
        stop_itr = [500000, ],
        total_epoch = 8000,
        eval_interval = 1000,
        dataloader = dict(
                type = 'build_simple_dataloader_DGL',
                batch_size = 256,
                number_workers = 0,
        ),
        eval_on = ['train', 'val', 'test'],
        max_ST_itr = 1,
)

extract_key_subgraph = dict(
        type = 'Extract_Syn_Multi_Process',
        # keep_subgraph_each_class = 5,
)
