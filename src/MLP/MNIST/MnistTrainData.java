package MLP.MNIST;

import Matrices.ActivationMatrix;

public class MnistTrainData extends ActivationMatrix {

    public MnistTrainData(MnistVector[] items) {
        super(784, items.length);

        int numberOfItems = items.length;
        double[][] data = this.getData();

        for(int i = 0; i < numberOfItems; i++) {
            double[][] vectorData = items[i].getData();
            for (int j = 0; j < 784; j++) {
                data[j][i] = vectorData[j][0];
            }
        }
    }

    /**
     * tronque la training data à i, pour éviter de faire exploser mon pauvre cpu
     * @param items
     * @param i
     */
    public MnistTrainData(MnistVector[] items, int i) {
        super(784, i);
        int numberOfItems = i;
        double[][] data = this.getData();

        for(int k = 0; k < numberOfItems; k++) {
            double[][] vectorData = items[i].getData();
            for (int j = 0; j < 784; j++) {
                data[j][k] = vectorData[j][0];
            }
        }
    }
}