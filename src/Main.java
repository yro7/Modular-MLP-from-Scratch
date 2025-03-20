import MLP.MLP;
import Matrices.*;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.IntStream;
import MLP.Pair;

import javax.swing.text.Position;

import static Function.ActivationFunction.*;
import static Function.LossFunction.MSE;

// TODO ROADMAP :
/**
 * Transposer les matrices d'activation pour que chaque "item" soit une rangée au lieu d'une colonne
 * java utilise le row major order donc pour des copies d'arrays ce serait plus opti
 *
 * Implémenter les Trainers, Optimizers, Data(TrainingData & TestData)
 *
 */
public class Main {
    public static void main(String[] args) {


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


    public static void save(){

        MLP mlp = MLP.builder(42)
                .setRandomSeed(420)
                .addLayer(32, ReLU)
                .addLayer(32, ReLU)
                .addLayer(16, ReLU)
                .addLayer(16, ReLU)
                .addLayer(8, ReLU)
                .addLayer(8, ReLU)
                .build();

        ActivationMatrix batchInput = new ActivationMatrix(creerTableau(42,1));
        ActivationMatrix batchTheorique = new ActivationMatrix(creerTableau(8,1));

        ActivationMatrix batchTheoriqueClone = batchTheorique.clone();
        ActivationMatrix batchInputClone = batchInput.clone();

        printLoss(mlp, batchInput, batchTheorique);

        mlp.getLayer(0).getWeightMatrix().print();

        for(int i = 0; i < 300_000; i++){
            //     mlp.getLayer(4).getBiasVector().print();
            mlp.updateParameters(batchInput, batchTheorique, MSE);
        }

        mlp.getLayer(0).getWeightMatrix().print();

        printLoss(mlp, batchInput, batchTheorique);

        assert(batchTheorique.equals(batchTheoriqueClone));
        assert(batchInputClone.equals(batchInput));

    }

}