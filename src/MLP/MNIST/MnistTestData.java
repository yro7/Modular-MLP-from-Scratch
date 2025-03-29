package MLP.MNIST;

import MLP.Data.LabeledTestData;

public class MnistTestData extends LabeledTestData {


    public MnistTestData(int size, int inputDimension, int outputDimension, String path, String labelPath) {
        super(size, inputDimension, outputDimension, path, labelPath);
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


