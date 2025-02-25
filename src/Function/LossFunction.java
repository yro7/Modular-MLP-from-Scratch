package Function;

import Matrices.ActivationMatrix;

import java.util.function.BiFunction;

public enum LossFunction {


    // TODO   mettre les bonnes dérivées
    /**
     * Mean Squared Error loss function (Erreur moyenne au carré).
     */
    MSE(
            (y_output,y_true) -> (y_output.substract(y_true).square().sum()) / y_output.size(),
            (y,y2) -> y.substract(y2).multiply(2).sum()
    ),

    /**
     * Mean Absolute Error loss function (Erreur moyenne absolue).
     */
    MAE(
            (y_output,y_true) -> Math.abs(y_output.substract(y_true).sum()),
            (y,y2) -> y.substract(y2).multiply(2).sum()

    ),

    /**
     * Log-cosh Loss (Erreur Log-cosh). Voir <a href="https://stats.stackexchange.com/questions/464354/when-is-log-cosh-loss-used#464374">When is Log-Cosh Loss Used? Stackoverflow.</a>
     */

    LogCosh(
            (y_output,y_true) -> y_output.substract(y_true).cosh().log().sum(),
            (y,y2) -> y.substract(y2).multiply(2).sum()
    );


    public final BiFunction<ActivationMatrix,ActivationMatrix, Double> lossFunction;
    public final BiFunction<ActivationMatrix,ActivationMatrix, Double> derivative;

    LossFunction(BiFunction <ActivationMatrix,ActivationMatrix,Double> lossFunction,
            BiFunction <ActivationMatrix,ActivationMatrix,Double> derivative){
        this.lossFunction = lossFunction;
        this.derivative = derivative;
    }

    public double apply(ActivationMatrix networkOutput, ActivationMatrix input) {
        return this.lossFunction.apply(networkOutput, input);
    }
}
