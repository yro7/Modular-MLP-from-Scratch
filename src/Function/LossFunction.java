package Function;

import Matrices.ActivationMatrix;
import Matrices.GradientMatrix;

import java.util.function.BiFunction;

public interface LossFunction extends BiFunction<ActivationMatrix, ActivationMatrix, Double> {

    public Double apply(ActivationMatrix y_pred, ActivationMatrix y_true);

    /**
     * Calcule la dérivée de la fonction de coût par rapport à la sortie attendue, à la sortie obtenue.
     *
     * @param y_pred la prédiction du réseau
     * @param y_true La matrice de sortie attendue
     * @return La matrice "output", modifiée
     * @mutable Cette méthode modifie la matrice output
     */
    public GradientMatrix applyDerivative(ActivationMatrix y_pred, ActivationMatrix y_true);

    final LossFunction MSE = new MSE();
    final LossFunction MAE = new MSE();

    /**
     * Log-cosh Loss (Erreur Log-cosh). Voir <a href="https://stats.stackexchange.com/questions/464354/when-is-log-cosh-loss-used#464374">When is Log-Cosh Loss Used? Stackoverflow.</a>
     */
    final LossFunction LogCosh = new LogCosh();

    /**
     * Binary Cross Entropy. Utilisé pour la classification binaire. Voir <a href="https://stats.stackexchange.com/questions/464354/when-is-log-cosh-loss-used#464374">When is Log-Cosh Loss Used? Stackoverflow.</a>
     */
    final LossFunction BCE = new BCE();


    final class MSE implements LossFunction {

        @Override
        public Double apply(ActivationMatrix y_pred, ActivationMatrix y_true) {
            return y_pred
                    .substract(y_true)
                    .square()
                    .sum() / y_pred.size();
        }

        @Override
        public GradientMatrix applyDerivative(ActivationMatrix output, ActivationMatrix expected) {
            return output.substract(expected)
                    .multiply(2)
                    .divide(output.size())
                    .toGradientMatrix();
        }
    }

    /**
     * Mean Absolute Error loss function (Erreur moyenne absolue).
     */
    final class MAE implements LossFunction {

        @Override
        public GradientMatrix applyDerivative(ActivationMatrix y_pred, ActivationMatrix y_true) {
            return y_pred.substract(y_true).sign().divide(y_true.getBatchSize()).toGradientMatrix();
        }

        @Override
        public Double apply(ActivationMatrix y_pred, ActivationMatrix y_true) {
            return Math.abs(y_pred.substract(y_true).sum());
        }
    }


    final class LogCosh implements LossFunction {

        @Override
        public GradientMatrix applyDerivative(ActivationMatrix y_pred, ActivationMatrix y_true) {
            return  y_pred.substract(y_true).tanh().divide(y_true.getBatchSize()).toGradientMatrix();
        }
        @Override
        public Double apply(ActivationMatrix y_pred, ActivationMatrix y_true) {
            return y_pred.substract(y_true).cosh().log().sum() / y_true.getBatchSize();
        }
    }

    // TODO opti à fond
    final class BCE implements LossFunction {

        @Override
        public GradientMatrix applyDerivative(ActivationMatrix y_pred, ActivationMatrix y_true) {
            ActivationMatrix y_pred2 = y_pred.clone().add(1.0).multiply(-1.0);
            ActivationMatrix y_true2 = y_true.clone().add(1.0).multiply(-1.0);
            return y_true.hadamardQuotient(y_pred)
                    .substract(y_true2.hadamardQuotient(y_pred2))
                    .divide(-1*(y_pred.size()))
                    .toGradientMatrix();
        }

        @Override
        public Double apply(ActivationMatrix y_pred, ActivationMatrix y_true) {
            ActivationMatrix y_pred2 = y_pred.clone().add(1.0).multiply(-1.0);
            ActivationMatrix y_true2 = y_true.clone().add(1.0).multiply(-1.0);

            return y_true.hadamardProduct(y_pred.log())
                    .add(y_true2.hadamardProduct(y_pred2.log())).sum() / -1.0*(y_pred.size());
        }
    }



}
