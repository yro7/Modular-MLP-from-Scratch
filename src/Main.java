import MLP.MLP;
import Matrices.*;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.IntStream;
import MLP.Pair;
import MLP.Trainer;

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

        Trainer trainer = Trainer.builder()
                .setLossFunction(MSE)
                .setOptimizer(null)
                .build();
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

    public static void printLoss(MLP mlp, ActivationMatrix batchInput, ActivationMatrix batchTheorique){
        System.out.println("Loss : " + mlp.computeLoss(batchInput, batchTheorique, BCE));
    }


    public static void save(){
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


    }

}