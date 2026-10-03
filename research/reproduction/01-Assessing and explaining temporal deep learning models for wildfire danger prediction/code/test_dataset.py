from datasets.dataset import FireDataset
import json


with open(
    "configs/config_lstm/config_train.json"
) as f:
    cfg = json.load(f)


dataset = FireDataset(
    **cfg["dataset"]["args"],
    dynamic_features=cfg["features"]["dynamic"],
    static_features=cfg["features"]["static"]
)


print("dataset length:", len(dataset))


sample = dataset[0]


print("======================")
print("dynamic shape:", sample[0].shape)
print("static shape:", sample[1].shape)
print("label:", sample[3])

print("======================")
print("dynamic:")
print(sample[0])

print("======================")
print("static:")
print(sample[1])