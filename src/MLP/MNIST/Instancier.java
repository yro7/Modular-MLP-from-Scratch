package MLP.MNIST;
import MLP.Pair;

import Function.ActivationFunction;
import MLP.MLP;
import Matrices.ActivationMatrix;

import java.io.IOException;
import java.util.List;

import static Function.ActivationFunction.*;
import static Function.LossFunction.*;

public class Instancier {

    public static void main(String[] args) throws IOException {
        // Chemins vers les fichiers MNIST
        String imagesPath = "src/MLP/MNIST/data/t10k-images.idx3-ubyte";
        String labelsPath = "src/MLP/MNIST/data/t10k-labels.idx1-ubyte";

        // Contient 10k MnistVector
        MnistVector[] mnistVectors = MnistDataReader.readData(imagesPath, labelsPath);

        int batchSize = 3000;

        MnistTrainData data = new MnistTrainData(mnistVectors, batchSize);

        // Construction du MLP
        MLP mnistMLP = MLP.builder(784)
                .setRandomSeed(2)
                .addLayer(256, ReLU)
                .addLayer(128, ReLU)
                .addLayer(10, SoftMax)
                .build();

        ActivationMatrix batchTheorique = new ActivationMatrix(batchSize, 10);

        // Construction de la matrice des sorties attendues
        for(int i = 0; i < batchSize; i++){
            MnistVector vectorI = mnistVectors[i];
            int label = vectorI.getLabel();

            for(int k = 0; k < 10; k++){
                batchTheorique.getData()[i][k] = (k == label) ? 1 : 0;
            }
        }

        // Calcul de la loss initiale
        printLoss(mnistMLP, data, batchTheorique);

         for(int i = 0; i < 10; i++){
         mnistMLP.updateParameters(data, batchTheorique, CE);
         System.out.println("Etape d'entraînement " + i + " finie!");
         printLoss(mnistMLP, data, batchTheorique);
         }

        for(int i = 0; i < 5; i++){
            MnistVector vector = mnistVectors[i];
            List<Pair<ActivationMatrix, ActivationMatrix>> res = mnistMLP.feedForward(vector);
            int pred = maxIndiceOfArray(res.getLast().getA().getData()[0]);
            System.out.println("Prédiction : " + pred + "   |   Valeur réelle : " + vector.getLabel());

        }

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
        System.out.println("Loss : " + mlp.computeLoss(batchInput, batchTheorique, CE));
    }

    public static int maxIndiceOfArray(double[] array){
        int res = 0;
        double max = array[0];
        for(int i = 0; i < array.length; i ++){
            if(max < array[i]) {
                res = i;
                max = array[i];
            }
        }

        return res;
    }
}