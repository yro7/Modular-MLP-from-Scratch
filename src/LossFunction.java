import java.util.function.BiFunction;

public enum LossFunction {

    /**
     * Mean Squared Error loss function (Erreur moyenne au carré).
     */
    MSE((y_output,y_true) -> Math.pow(y_output.substract(y_true).sum(),2)),

    /**
     * Mean Absolute Error loss function (Erreur moyenne absolue).
     */
    MAE( (y_output,y_true) -> Math.abs(y_output.substract(y_true).sum())),

    /**
     * Log-cosh Loss (Erreur Log-cosh). Voir <a href="https://stats.stackexchange.com/questions/464354/when-is-log-cosh-loss-used#464374">When is Log-Cosh Loss Used? Stackoverflow.</a>
     */

    LogCosh( (y_output,y_true) -> y_output.substract(y_true) ,


    public final BiFunction<ActivationVector,ActivationVector, Double> lossFunction;

    LossFunction(BiFunction <ActivationVector,ActivationVector,Double> lossFunction){
        this.lossFunction = lossFunction;
    }
}
