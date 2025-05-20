package mlps.mnist;

import mlps.Classification;
import mlps.data.LabeledDataset;
import mlps.data.Loaders.Dataloader;

public class MnistDataset extends LabeledDataset<MnistImage, Classification> {


    public static final String TRAIN_IMAGES_PATH = "src/mlps/mnist/data/train-images.idx3-ubyte";
    public static final String TRAIN_LABELS_PATH = "src/mlps/mnist/data/train-labels.idx1-ubyte";

    public static final String TEST_IMAGES_PATH = "src/mlps/mnist/data/t10k-images.idx3-ubyte";
    public static final String TEST_LABELS_PATH = "src/mlps/mnist/data/t10k-labels.idx1-ubyte";

    public MnistDataset() {
        Dataloader<MnistImage, Classification> train =
                new MnistDataloader(TRAIN_IMAGES_PATH, TRAIN_LABELS_PATH, 0, 60_000);
        Dataloader<MnistImage, Classification> test =
                new MnistDataloader(TEST_IMAGES_PATH, TEST_LABELS_PATH, 0, 10_000);

        super(test, train);
    }
}
