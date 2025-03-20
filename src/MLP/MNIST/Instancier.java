package MLP.MNIST;

import java.io.IOException;

public class Instancier {

    public static void main(String[] args) throws IOException {

        // Contient 10k MnistMatrix
        MnistVector[] mnistVectors = MnistDataReader.readData("src/MLP/MNIST/data/t10k-images.idx3-ubyte",
                "src/MLP/MNIST/data/t10k-labels.idx1-ubyte");

        MnistTrainData data = new MnistTrainData(mnistVectors);

        data.printDimensions("gouga");


    }

    private static void printMnistMatrix(final MnistMatrix matrix) {
        System.out.println("label: " + matrix.getLabel());
        for (int r = 0; r < matrix.getNumberOfRows(); r++ ) {
            for (int c = 0; c < matrix.getNumberOfColumns(); c++) {
                System.out.print(matrix.getValue(r, c) + " ");
            }
            System.out.println();
        }
    }
}