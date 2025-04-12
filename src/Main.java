import MLP.MLP;
import MLP.Optimizers.SGD;
import MLP.Regularizations.ParameterRegularization;
import Matrices.*;

import static Function.ActivationFunction.*;
import static Function.LossFunction.*;

// TODO ROADMAP :
/**
 * Implémenter ADAM & autres optimizers
 * Implémenter régularization
 * Implémenter dropout
 */
public class Main {
    public static void main(String[] args) {

        MLP mlp = MLP.builder(2)
                .setRandomSeed(3)
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

        for(int i = 0; i < 10000; i++){
            mlp.updateParameters(batchInput, batchTheorique, MSE, new SGD(0.1), new ParameterRegularization.L1(0.1));
            if(i % 100 == 0) printLoss(mlp, batchInput, batchTheorique);
        }

    }
        public static void printLoss(MLP mlp, ActivationMatrix batchInput, ActivationMatrix batchTheorique){
        System.out.println("Loss : " + mlp.computeLoss(batchInput, batchTheorique, MSE));
    }

}