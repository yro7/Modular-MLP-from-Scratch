package MLP.MNIST;
import MLP.Pair;

import Function.ActivationFunction;
import MLP.MLP;
import Matrices.ActivationMatrix;

import java.io.IOException;
import java.util.List;

import static Function.ActivationFunction.ReLU;
import static Function.ActivationFunction.Sigmoid;
import static Function.LossFunction.MSE;

public class Instancier {

    public static void main(String[] args) throws IOException {

        // Contient 10k MnistMatrix
        MnistVector[] mnistVectors = MnistDataReader.readData("src/MLP/MNIST/data/t10k-images.idx3-ubyte",
                "src/MLP/MNIST/data/t10k-labels.idx1-ubyte");

        final int batchSize = 3000;
        MnistTrainData data = new MnistTrainData(mnistVectors, batchSize);

        data.printDimensions("test");

        System.out.println(mnistVectors.length + "caca");


        MLP mnistMLP = MLP.builder(784)
                .setRandomSeed(2)
                .addLayer(256, ReLU)
                .addLayer(128, ReLU)
                .addLayer(10, Sigmoid)
                .build();

        //List<Pair<ActivationMatrix, ActivationMatrix>> res = mnistMLP.feedForward(data);

        ActivationMatrix batchTheorique = new ActivationMatrix(10, batchSize);

        // matrice d'activation, vecturs en hauteur de taille 10
        // et 3000 vecteurs comme ça, donc 10k colonnes

        for(int i = 0; i < batchSize; i++){
            MnistVector vectorI = mnistVectors[i];
            int label = vectorI.getLabel();

            for(int k = 0; k < 10; k++){
                batchTheorique.getData()[k][i] = (k == label) ? 1 : 0;
            }
        }

        printLoss(mnistMLP, data, batchTheorique);

        for(int i = 0; i < 20; i++){
            mnistMLP.updateParameters(data, batchTheorique, MSE);
            System.out.println("Etape d'entraînement " + i + " finie!");
            printLoss(mnistMLP, data, batchTheorique);


        }

        printLoss(mnistMLP, data, batchTheorique);

       // res.getLast().getA().printDimensions("c");

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

    public static void printLoss(MLP mlp, ActivationMatrix batchInput, ActivationMatrix batchTheorique){
        System.out.println("Loss : " + mlp.computeLoss(batchInput, batchTheorique, MSE));
    }
}