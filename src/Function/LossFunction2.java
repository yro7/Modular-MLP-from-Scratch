package Function;

import Matrices.ActivationMatrix;
import Matrices.GradientMatrix;

import java.util.function.BiFunction;

public interface LossFunction2 extends BiFunction<ActivationMatrix, ActivationMatrix, Double> {


    // TODO try both interface/enum approaches to see which one is the best (+ finishing implementing with interfaces)
    public GradientMatrix applyDerivative(ActivationMatrix output, ActivationMatrix expected);

    LossFunction2 MSE = new MSE();


    final class MSE implements LossFunction2 {

        @Override
        public Double apply(ActivationMatrix output, ActivationMatrix expected) {
            return output
                    .substract(expected)
                    .square()
                    .sum() / output.size();
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
    final class MAE implements LossFunction2 {

        @Override
        public GradientMatrix applyDerivative(ActivationMatrix output, ActivationMatrix expected) {
            return output.substract(expected).toGradientMatrix();
        }

        @Override
        public Double apply(ActivationMatrix output, ActivationMatrix expected) {
            return Math.abs(output.substract(expected).sum());
        }
    }

}
