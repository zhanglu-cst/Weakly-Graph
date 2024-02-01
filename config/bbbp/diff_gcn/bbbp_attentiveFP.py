data_filename = 'bbbp.json'

work_dir = './work_dir/' + data_filename.split('.')[0]

total_number_itr = 10
number_classes = 2

dataset = dict(
        train = dict(
                type = 'Moleculars',
                filename = data_filename,
                split = 'train',
                token_func = 'func_group'
        ),
        val = dict(
                type = 'Moleculars',
                filename = data_filename,
                split = 'val',
                token_func = 'func_group'
        ),
        test = dict(
                type = 'Moleculars',
                filename = data_filename,
                split = 'test',
                token_func = 'func_group'
        )
)

key_subgraph = dict(
        type = 'Key_SubGraph_Mol',
        init_filename = 'bbbp_5.json',
        init_key_score = 100,
        incremental = True,
        overwrite_confict = True,
        max_capacity_each_class = 500,
)

pseudo_label = dict(
        type = 'Assign_Voter_Mol',
        number_classes = number_classes,
)

classifier = dict(
        type = 'AttentiveFPGNN',
        num_timesteps = 2,
        node_feat_size = 39,
        edge_feat_size = 10,
        num_layers = 2,
        graph_feat_size = 200,
        dropout = 0,
        output_dim = 1,
)

trainer_classifier = dict(
        optimizer_classifier = dict(
                name = 'AdamW',
                lr = 0.001,
        ),
        train_val_dataset = dict(
                type = 'Mol_Dataset',
                max_upsample = 1,
        ),
        stop_itr = [5000, ],
        total_epoch = 20000,
        eval_interval = 50,
        dataloader = dict(
                type = 'build_simple_dataloader_mol',
                batch_size = 64,
        ),
        eval_on = ['train', 'val', 'test'],
        eval_thr = 0.5
)

token = dict(
        type = 'Token_Moleculars_Functional_Group',
        max_frag_number = 8,
        load_dump = False,
)

extract_key_subgraph = dict(
        type = 'Updater_TF_IDF_Mol',
        IDF_power = 4,
        number_classes = number_classes,
        keep_tokens = 500,
)

logger = dict(
        enable_wandb = True,
        entity = 'weakly_graph',
        project = data_filename,
        name = 'run_good',
)

init_key = dict(
        precision_thr = [0.99, 0.99]
)
