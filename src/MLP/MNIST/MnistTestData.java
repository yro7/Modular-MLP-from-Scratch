package MLP.MNIST;

import MLP.Classification;
import MLP.Data.LabeledTestData;

import java.util.ArrayList;
import java.util.List;

public class MnistTestData extends LabeledTestData {

    public static final String IMAGES_PATH = "src/MLP/MNIST/data/train-images.idx3-ubyte";
    public static final String LABELS_PATH = "src/MLP/MNIST/data/train-labels.idx1-ubyte";
    public static final int SIZE = 60_000;
    public static final int OUTPUT_DIMENSION = 10;
    public static final int INPUT_DIMENSION = 784;

    private List<LabeledDataSample<MnistImage, Classification>> cachedData;
    private boolean isDataLoaded = false;


    public MnistTestData() {
        super(SIZE, INPUT_DIMENSION, OUTPUT_DIMENSION, IMAGES_PATH, LABELS_PATH);
    }

    @Override
    public double[] vectorizeInput(Object obj) {
        return new double[0];
    }

    @Override
    public double[] vectorizeOutput(Object input) {
        return new double[0];
    }

    @Override
    public LabeledDataSample load(int i) {
        return null;
    }
}


