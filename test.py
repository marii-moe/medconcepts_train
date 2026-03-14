from medconcepts_train import load_environment

env = load_environment()
dataset = env.dataset

for row in dataset.select(range(5)):
    print(row)
    print()
