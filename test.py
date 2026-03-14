from medconcepts_train import load_environment
import verifiers as vf

env_id = 'medconcepts_train'

env = vf.load_environment(env_id)

print(env)

train_env_group = vf.EnvGroup(
    envs=[env],
    env_names=[env_id],
    map_kwargs=dict(writer_batch_size=1),  # set defensively to not error on map operations on large datasets
)

train_dataset = train_env_group.get_dataset()

for row in train_dataset.select(range(5)):
    print(row)
    print()