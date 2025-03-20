package MLP.MNIST;

import Matrices.ActivationMatrix;

public class MnistTrainData extends ActivationMatrix {

    public MnistTrainData(MnistVector[] items) {
        super(784, items.length);

        int numberOfItems = items.length;
        System.out.println("number of items : " + numberOfItems);

        // Accès direct aux tableaux pour maximiser la performance
        double[][] data = this.getData();

        for(int i = 0; i < numberOfItems; i++) {
            double[][] vectorData = items[i].getData();
            for (int j = 0; j < 784; j++) {
                data[j][i] = vectorData[j][0];
            }

            items[i].printDimensions("Item " + i);
        }
    }
}