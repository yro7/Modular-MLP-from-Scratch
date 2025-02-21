import Function.ActivationFunction;
import Matrices.ActivationMatrix;
import Matrices.BiasVector;
import Matrices.Matrix;
import Matrices.WeightMatrix;

import java.util.Arrays;

public class Main {
    public static void main(String[] args) {


        MLP mlp = MLP.builder(4)
                .addLayer(2, ActivationFunction.ReLU)
                .addLayer(4, ActivationFunction.Sigmoid)
                .addLayer(2, ActivationFunction.Sigmoid)
                .build();


        ActivationMatrix am = new ActivationMatrix(creerTableau(4,1));

        mlp.feedForward(am);

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