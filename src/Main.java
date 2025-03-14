import MLP.MLP;
import Matrices.*;

import java.util.List;
import java.util.stream.IntStream;
import MLP.Pair;

import javax.swing.text.Position;

import static Function.ActivationFunction.ReLU;
import static Function.ActivationFunction.Sigmoid;
import static Function.LossFunction.MSE;

public class Main {
    public static void main(String[] args) {


        MLP mlp = MLP.builder(43)
                .setRandomSeed(2)
                .addLayer(7, ReLU)
                .addLayer(3, ReLU)
                .addLayer(17, ReLU)
                .addLayer(24, ReLU)
                .addLayer(24, ReLU)
                .addLayer(4, ReLU)
                .addLayer(8, Sigmoid)
                .build();

        ActivationMatrix batchInput = new ActivationMatrix(creerTableau(43,1));
        ActivationMatrix batchTheorique = new ActivationMatrix(creerTableau(8,1));

        printLoss(mlp, batchInput, batchTheorique);
        loopBackpro(mlp, batchInput, batchTheorique, 500);
        printLoss(mlp, batchInput, batchTheorique);


        //mlp.backpropagate(batchInput, batchTheorique, MSE);
        /**
        IntStream.range(1,15).forEach(i -> {
            mlp.backpropagate(batchInput, batchTheorique, MSE);
            loss[0] = mlp.computeLoss(batchInput, batchTheorique, MSE);
            System.out.println("loss at iteration : " + i + " : "  + loss[0]);
**/

    }

    public static double[][] creerTableau(int n, int p){
        double[][] res = new double[n][p];
        int compteur = 1;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < p; j++) {
                res[i][j] = compteur++;
            }
        }
        return res;
    }

    public static void loopBackpro(MLP mlp, ActivationMatrix batchInput, ActivationMatrix batchTheorique, int n){
        IntStream.range(1,n).forEach(i -> {
            mlp.updateParameters(batchInput, batchTheorique, MSE);
    });
    }

    public static void printLoss(MLP mlp, ActivationMatrix batchInput, ActivationMatrix batchTheorique){
        System.out.println("Loss : " + mlp.computeLoss(batchInput, batchTheorique, MSE));
    }


}