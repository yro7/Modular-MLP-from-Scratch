import MLP.MLP;
import Matrices.*;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.IntStream;
import MLP.Pair;

import javax.swing.text.Position;

import static Function.ActivationFunction.*;
import static Function.LossFunction.*;

// TODO ROADMAP :
/**
 * Implémenter les Trainers, Optimizers, Data(TrainingData & TestData)
 *
 */
public class Main {
    public static void main(String[] args) {

        MLP mlp = MLP.builder(2)
                .setRandomSeed(69)
                .addLayer(4, ReLU)
                .addLayer(1, Sigmoid)
                .build();

        double[][] xorData = {
                {0, 0},
                {0, 1},
                {1, 0},
                {1, 1}
        };

        double[][] xorResult = {
                {0},
                {1},
                {1},
                {0},
        };


        ActivationMatrix batchInput = new ActivationMatrix(xorData);
        ActivationMatrix batchTheorique = new ActivationMatrix(xorResult);


        loopBackpro(mlp, batchInput, batchTheorique, 10_000);



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
            mlp.updateParameters(batchInput, batchTheorique, BCE);
            if(i % 100 == 0) {
                System.out.print(" loss étape " + i);
                printLoss(mlp, batchInput, batchTheorique);

            }

    });
        printLoss(mlp, batchInput, batchTheorique);

    }

    public static void printLoss(MLP mlp, ActivationMatrix batchInput, ActivationMatrix batchTheorique){
        System.out.println("Loss : " + mlp.computeLoss(batchInput, batchTheorique, BCE));
    }


    public static void save(){


    }

}