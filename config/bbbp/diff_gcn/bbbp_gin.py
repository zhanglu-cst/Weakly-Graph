# _base_ = './default.py'

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
        init_filename = data_filename,
)

pseudo_label = dict(
        type = 'Assign_Voter_Mol',
        number_classes = number_classes,
)

classifier = dict(
        type = 'GIN',
        num_node_emb_list = [39],
        num_edge_emb_list = [10],
        num_layers = 3,
        emb_dim = 256,
        readout = 'sum',
        JK = 'last',
        dropout = 0.1,
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
        stop_itr = [500000],
        total_epoch = 10000,
        eval_interval = 100,

        dataloader = dict(
                type = 'build_simple_dataloader',
                batch_size = 128,
        ),
        eval_on = ['train', 'val', 'test'],
        eval_thr = 0.5
)

token = dict(
        type = 'Token_Moleculars_Functional_Group',
        max_frag_number = 8,
        load_dump = False,
)

update_key_subgraph = dict(
        type = 'Updater_TF_IDF_Mol',
        IDF_power = 6,
        number_classes = number_classes,
        keep_tokens = 500,
)

logger = dict(
        enable_wandb = True,
        entity = 'weakly_graph',
        project = data_filename,
        name = 'run_good',
)
