import Function.ActivationFunction;
import Matrices.*;

import java.util.List;
import java.util.Random;

import static Function.ActivationFunction.ReLU;
import static Function.LossFunction.MSE;

public class Main {
    public static void main(String[] args) {


        MLP mlp = MLP.builder(43)
                .setRandomSeed(2)
                .addLayer(2, ReLU)
                .addLayer(4, ActivationFunction.Sigmoid)
                .addLayer(2, ActivationFunction.Sigmoid)
                .build();

        mlp.print();

        ActivationMatrix aa = new ActivationMatrix(creerTableau(43,1));
        ActivationMatrix bb = new ActivationMatrix(creerTableau(43,1));

        List<GradientMatrix> cc = mlp.gradientDescent(aa,bb,MSE);
        cc.forEach(gm -> {
            System.out.println("cacagougouga");
            gm.print();
        });
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