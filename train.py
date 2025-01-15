from training.behavioural_cloning import behavioural_cloning_train

def main():
    print("===Training Find wood model===")
    behavioural_cloning_train(
        data_dir="data/MirroredMineRLBasaltFindWood-v0",
        in_model="data/VPT-models/foundation-model-1x.model",
        in_weights="./train/ppo_updated_weights_rmspropGR2.weights",
        out_weights="./SecondTrainMirroredVids.weights"
    )

if __name__ == "__main__":
    main()

''' per provare ad addestrare sui video tagliati
behavioural_cloning_train(
        data_dir="data/CuttedVideos",
        in_model="data/VPT-models/foundation-model-1x.model",
        in_weights="data/VPT-models/foundation-model-1x.weights",
        out_weights="train/CuttedVideos.weights"
    )
'''