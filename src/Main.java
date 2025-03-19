import MLP.MLP;
import Matrices.*;

import java.util.ArrayList;
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
                .setRandomSeed(420)
                .addLayer(7, ReLU)
                .addLayer(7, ReLU)
                .addLayer(7, ReLU)
                .addLayer(7, ReLU)
                .addLayer(8, ReLU)
                .build();

        ActivationMatrix batchInput = new ActivationMatrix(creerTableau(43,1));
        ActivationMatrix batchTheorique = new ActivationMatrix(creerTableau(8,1));


        for(int i = 0; i < 50; i++){
       //     mlp.getLayer(4).getBiasVector().print();
            mlp.updateParameters(batchInput, batchTheorique, MSE);
        }
        printLoss(mlp, batchInput, batchTheorique);


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
        printLoss(mlp, batchInput, batchTheorique);
        IntStream.range(1,n).forEach(i -> {
            mlp.updateParameters(batchInput, batchTheorique, MSE);
    });
        printLoss(mlp, batchInput, batchTheorique);

    }

    public static void printLoss(MLP mlp, ActivationMatrix batchInput, ActivationMatrix batchTheorique){
        System.out.println("Loss : " + mlp.computeLoss(batchInput, batchTheorique, MSE));
    }


}