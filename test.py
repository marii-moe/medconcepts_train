from medconcepts_train import load_environment
import verifiers as vf

env_ids = ["maziyar/OpenMed_ICD10", "medconcepts_train"]

envs = [vf.load_environment(env_ids[0]), vf.load_environment(env_ids[1])]

names = ["openmed-icd10", "medconcepts_train"]


train_env_group = vf.EnvGroup(
    envs=envs,
    env_names=names,
    map_kwargs=dict(writer_batch_size=1),  # set defensively to not error on map operations on large datasets
)

train_dataset = train_env_group.get_dataset()

for row in train_dataset.select(range(5)):
    print(row['task'])
    print()

print(set(train_dataset["task"]))

print(set(names))

assert set(train_dataset["task"]) == set(names)