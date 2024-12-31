from behavioural_cloning import behavioural_cloning_train

def main():
    print("===Training FindCave model===")
    behavioural_cloning_train(
        data_dir="data/MineRLBasaltFindWood-v0",
        in_model="data/VPT-models/foundation-model-1x.model",
        in_weights="data/VPT-models/foundation-model-1x.weights",
        out_weights="train/MineRLBasaltFindWood.weights"
    )

if __name__ == "__main__":
    main()
