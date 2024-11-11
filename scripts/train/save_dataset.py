from datasets import load_dataset


dataset = load_dataset("cerebras/SlimPajama-627B")
dataset.save_to_disk('/home/user1/workspace/datasets/SlimPajama-627B')

