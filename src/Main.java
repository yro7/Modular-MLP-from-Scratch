import Function.ActivationFunction;
import Matrices.ActivationMatrix;
import Matrices.BiasVector;
import Matrices.Matrix;
import Matrices.WeightMatrix;

import java.util.Arrays;

public class Main {
    public static void main(String[] args) {


        MLP mlp = MLP.builder(43)
                .addLayer(2, ActivationFunction.ReLU)
                .addLayer(4, ActivationFunction.Sigmoid)
                .addLayer(123, ActivationFunction.Sigmoid)
                .addLayer(4, ActivationFunction.Sigmoid)
                .addLayer(43, ActivationFunction.Sigmoid)
                .addLayer(43, ActivationFunction.Sigmoid)
                .addLayer(4, ActivationFunction.Sigmoid)
                .addLayer(43, ActivationFunction.Sigmoid)
                .addLayer(4, ActivationFunction.Sigmoid)
                .addLayer(43, ActivationFunction.Sigmoid)
                .addLayer(4, ActivationFunction.Sigmoid)
                .addLayer(4, ActivationFunction.Sigmoid)
                .addLayer(2, ActivationFunction.Sigmoid)
                .build();

        ActivationMatrix am = new ActivationMatrix(creerTableau(43,3));

        ActivationMatrix result = mlp.feedForward(am);

        result.print();

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


}