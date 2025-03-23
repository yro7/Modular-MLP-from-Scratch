package MLP.MNIST;

import Matrices.ActivationMatrix;

public class MnistTrainData extends ActivationMatrix {

    public MnistTrainData(MnistVector[] items) {
        super(items.length, 784);

        int numberOfItems = items.length;
        double[][] data = this.getData();

        for(int i = 0; i < numberOfItems; i++) {
            double[][] vectorData = items[i].getData();
            for (int j = 0; j < 784; j++) {
                data[i][j] = vectorData[j][0];
            }
        }
    }

    /**
     * tronque la training data à numItems, pour éviter de faire exploser mon pauvre cpu
     */
    public MnistTrainData(MnistVector[] vectors, int batchSize) {
        super(batchSize, 784);

        // Remplir la matrice avec les données des vecteurs MNIST
        for (int i = 0; i < batchSize; i++) {
            double[] values = vectors[i].getData()[0];
            for (int j = 0; j < values.length; j++) {
                this.getData()[i][j] = values[j];
            }
        }
    }


}