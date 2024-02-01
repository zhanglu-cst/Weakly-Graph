import wandb

wandb.init(project = "test_new", entity = "weakly_graph", name = 'hello_1000', config = {
    "learning_rate": 0.001,
    "epochs": 100,
    "batch_size": 128
})

for i in range(10):
    wandb.log({"loss": i + 2, 'itr': i * 10})
