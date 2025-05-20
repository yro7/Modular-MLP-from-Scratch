package mlps.data;

import mlps.data.Loaders.Dataloader;

/**
 * Représente un ensemble de données labelisées séparé en une
 * partie d'entraînement et une partie de test.
 * @param <InputType>
 * @param <OutputType>
 */
public class LabeledDataset<InputType, OutputType> {

    public Dataloader<InputType, OutputType> trainDataLoader;
    public Dataloader<InputType, OutputType> testDataloader;

    public LabeledDataset(Dataloader<InputType, OutputType> testDataloader, Dataloader<InputType, OutputType> trainDataLoader) {
        this.testDataloader = testDataloader;
        this.trainDataLoader = trainDataLoader;
    }


}
