package Function;

import Matrices.ActivationMatrix;
import Matrices.GradientMatrix;

import java.util.function.BiFunction;

public enum LossFunction implements BiFunction<ActivationMatrix, ActivationMatrix, Double> {


    /**
     * Mean Squared Error loss function (Erreur moyenne au carré).
     */
    MSE((y_output,y_true) -> y_output.substract(y_true)
                    .square()
                    .sum() / y_output.getBatchSize(), // On divise à la fin pr + d'efficacité de calcul (1 division au lieu de nxp)

            (y,y_true) -> y.substract(y_true)
                    .multiply(2)
                    .divide(y.getBatchSize())
                    .toGradientMatrix()
    ),

    /**
     * Mean Absolute Error loss function (Erreur moyenne absolue).
     */
    MAE((y_output,y_true) -> Math.abs(y_output.substract(y_true)
                    .sum()) / y_output.getBatchSize(),

            (y,y_true) -> y.substract(y_true)
                    .sign()
                    .divide(y_true.getBatchSize())
                    .toGradientMatrix()
    ),

    /**
     * Log-cosh Loss (Erreur Log-cosh). Voir <a href="https://stats.stackexchange.com/questions/464354/when-is-log-cosh-loss-used#464374">When is Log-Cosh Loss Used? Stackoverflow.</a>
     */

    LogCosh((y_output,y_true) -> y_output.substract(y_true)
                    .cosh()
                    .log()
                    .sum() / y_true.getBatchSize(),

            (y_output,y_true) -> y_output.substract(y_true)
                    .tanh()
                    .divide(y_true.getBatchSize())
                    .toGradientMatrix()
    );


    public final BiFunction<ActivationMatrix,ActivationMatrix, Double> lossFunction;
    public final BiFunction<ActivationMatrix,ActivationMatrix, GradientMatrix> derivative;

    LossFunction(BiFunction <ActivationMatrix,ActivationMatrix,Double> lossFunction,
            BiFunction <ActivationMatrix,ActivationMatrix,GradientMatrix> derivative){
        this.lossFunction = lossFunction;
        this.derivative = derivative;
    }

    public Double apply(ActivationMatrix networkOutput, ActivationMatrix input) {
        return this.lossFunction.apply(networkOutput, input);
    }
    /**
     * Calcule la dérivée de la fonction de coût par rapport à la sortie attendue, à la sortie obtenue.
     *
     * @param output la prédiction du réseau
     * @param expected La matrice de sortie attendue
     * @return La matrice "output", modifiée
     * @mutable Cette méthode modifie la matrice output
     */
    public GradientMatrix applyDerivative(ActivationMatrix output, ActivationMatrix expected) {
        return this.derivative.apply(output, expected);
    }
}
