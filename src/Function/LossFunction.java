package Function;

import Matrices.ActivationMatrix;
import Matrices.GradientMatrix;

import java.util.function.BiFunction;

public interface LossFunction extends BiFunction<ActivationMatrix, ActivationMatrix, Double> {

    double epsilon = 1e-12; // Delta pour éviter les divisions / log de 0
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

    /**
     * Mean Squared Error (Erreur quadratique)
     */
    final LossFunction MSE = new MSE();

    /**
     * Mean Absolute Error (Erreur absolue moyenne)
     */
    final LossFunction MAE = new MAE();

    /**
     * Log-cosh Loss (Erreur Log-cosh). Voir <a href="https://stats.stackexchange.com/questions/464354/when-is-log-cosh-loss-used#464374">When is Log-Cosh Loss Used? Stackoverflow.</a>
     */
    final LossFunction LogCosh = new LogCosh();

    /**
     * Binary Cross Entropy. Utilisé pour la classification binaire.
     **/
    final LossFunction BCE = new BCE();


    /**
     * Cross Entropy. Utilisé pour la classification multi-classes.
     **/
    final LossFunction CE = new CE();


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
                    .divide(output.getBatchSize())
                    .toGradientMatrix();
        }
    }

    /**
     * Mean Absolute Error loss function (Erreur moyenne absolue).
     */
    final class MAE implements LossFunction {

        @Override
        public GradientMatrix applyDerivative(ActivationMatrix y_pred, ActivationMatrix y_true) {
            return y_pred.substract(y_true)
                    .sign()
                    .divide(y_true.getBatchSize())
                    .toGradientMatrix();
        }

        @Override
        public Double apply(ActivationMatrix y_pred, ActivationMatrix y_true) {
            return Math.abs(y_pred.substract(y_true).sum()) / y_pred.size();
        }
    }


    final class LogCosh implements LossFunction {

        @Override
        public GradientMatrix applyDerivative(ActivationMatrix y_pred, ActivationMatrix y_true) {
            return  y_pred.substract(y_true)
                    .tanh()
                    .divide(y_true.getBatchSize())
                    .toGradientMatrix();
        }
        @Override
        public Double apply(ActivationMatrix y_pred, ActivationMatrix y_true) {
            return y_pred.substract(y_true).cosh().log().sum() / y_true.size();
        }
    }

    final class BCE implements LossFunction {

        @Override
        public GradientMatrix applyDerivative(ActivationMatrix y_pred, ActivationMatrix y_true) {
            GradientMatrix res = new GradientMatrix(y_pred.getNumberOfRows(), y_pred.getNumberOfColumns());
            res.applyToElements((i,j) -> {
                        double p = y_pred.getData()[i][j];
                        double y = y_true.getData()[i][j];
                        res.getData()[i][j] = (-y/p + (1-y)/(1-p))/4.0;
                    });
            return res;
        }

        @Override
        public Double apply(ActivationMatrix y_pred, ActivationMatrix y_true) {
            final double[] res = {0.0};
            y_pred.applyToElements((i,j) -> {
                double p = y_pred.getData()[i][j];
                double y = y_true.getData()[i][j];
                res[0] += y * Math.log(p+epsilon) + (1 - y) * Math.log(1 - p+epsilon);
            });
            return res[0] / -4.0;
        }
    }

        final class CE implements LossFunction {

            @Override
            public Double apply(ActivationMatrix y_pred, ActivationMatrix y_true) {
                double res = 0;
                for (int i = 0; i < y_true.getNumberOfRows(); i++) { // batch size
                    for (int j = 0; j < y_true.getNumberOfColumns(); j++) { // nombre de colonnes
                        if (y_true.getData()[i][j] > 0) {
                            res += y_true.getData()[i][j] * Math.log(y_pred.getData()[i][j] + epsilon);
                        }
                    }
                }
                res /= -1*y_pred.getBatchSize();
                return res;
            }

            @Override
            public GradientMatrix applyDerivative(ActivationMatrix y_pred, ActivationMatrix y_true) {
                return y_pred.substract(y_true).toGradientMatrix();
              /**  ActivationMatrix result = y_pred.clone();
                // Initialiser tous les éléments à 0
                for (int i = 0; i < result.getData().length; i++) {
                    for (int j = 0; j < result.getData()[i].length; j++) {
                        result.getData()[i][j] = 0;
                    }
                }

                // Calculer -y_true/y_pred/batchSize seulement où y_true > 0
                for (int i = 0; i < y_true.getData().length; i++) {
                    for (int j = 0; j < y_true.getData()[i].length; j++) {
                        if (y_true.getData()[i][j] > 0) {
                            result.getData()[i][j] = (-y_true.getData()[i][j] / y_pred.getData()[i][j]) / y_true.getBatchSize();
                        }
                    }
                }

                return result.toGradientMatrix();**/
            }
        }



}
