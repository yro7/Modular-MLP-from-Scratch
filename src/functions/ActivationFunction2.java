package functions;

import matrices.ActivationMatrix;

import java.util.function.Function;

import static functions.InitializationFunction.*;

/**
 * Représente la Fonction d'Activation utilisée par une couche du réseau de neurones.
 * Les fonctions d'activation sont des fonctions qui prennent une matrice d'activation
 * en entrée et la modifient pour obtenir la matrice post-activation fonction.
 *
 * @mutable Modifie la matrice sur laquelle est appliquée la fonction
 */
public interface ActivationFunction2 extends Function<ActivationMatrix, ActivationMatrix> {


    /**
     * Applique la fonction d'activation à une matrice d'activations.
     * @mutable Modifie la matrice passée en argument
     * @intermédiaire Retourne la matrice modifiée pour permettre le chaînage
     */
    ActivationMatrix apply(ActivationMatrix input);

    /**
     * Applique la dérivée de la fonction d'activation à une matrice d'activations.
     * @mutable Modifie la matrice passée en argument
     * @intermédiaire Retourne la matrice modifiée pour permettre le chaînage
     */
    ActivationMatrix applyDerivative(ActivationMatrix input);

    /**
     * Retourne la fonction d'initialization optimale correspondant à la fonction d'activation.
     */
    InitializationFunction getInitializationFunction();

    ActivationFunction2 ReLU = new ReLU();
    ActivationFunction2 TanH = new TanH();
    ActivationFunction2 Sigmoid = new Sigmoid();
    ActivationFunction2 Identity = new Identity();
    ActivationFunction2 SoftMax = new SoftMax();



    final class ReLU implements ActivationFunction2 {

        @Override
        public ActivationMatrix apply(ActivationMatrix input) {
            return input.applyFunction(d -> Math.max(0, d));
        }

        @Override
        public ActivationMatrix applyDerivative(ActivationMatrix input) {
            return input.applyFunction(d -> (d > 0.0) ? 1.0 : 0.0);
        }

        @Override
        public InitializationFunction getInitializationFunction(){
            return He;
        }
    }

    final class TanH implements ActivationFunction2 {

        @Override
        public ActivationMatrix apply(ActivationMatrix input) {
            return input.applyFunction(Math::tanh);
        }

        @Override
        public ActivationMatrix applyDerivative(ActivationMatrix input) {
            return input.applyFunction(d -> 1 - Math.pow(Math.tanh(d),2));
        }

        @Override
        public InitializationFunction getInitializationFunction() {
            return LeCun;
        }
    }

    final class Sigmoid implements ActivationFunction2 {

        @Override
        public ActivationMatrix apply(ActivationMatrix input) {
            return input.applyFunction(ActivationFunction::sigma);
        }

        @Override
        public ActivationMatrix applyDerivative(ActivationMatrix input) {
            return input.applyFunction(d -> sigma(d) * (1 - sigma(d)));
        }

        @Override
        public InitializationFunction getInitializationFunction() {
            return LeCun;
        }
    }

    final class Identity implements ActivationFunction2 {

        @Override
        public ActivationMatrix apply(ActivationMatrix input) {
            return input;
        }

        @Override
        public ActivationMatrix applyDerivative(ActivationMatrix input) {
            return input.applyFunction(d -> 1.0);
        }

        @Override
        public InitializationFunction getInitializationFunction() {
            return LeCun;
        }
    }

    final class SoftMax implements ActivationFunction2 {

        @Override
        public ActivationMatrix apply(ActivationMatrix input) {

            double[] sumsOverRows = input.sumOverRows();
            return input;
        }

        @Override
        public ActivationMatrix applyDerivative(ActivationMatrix input) {
            return input.applyFunction(d -> 1.0);
        }

        @Override
        public InitializationFunction getInitializationFunction() {
            return LeCun;
        }
    }

    /**
     * Renvoie la sigmoïde de z, càd
     * 1 / (    1 + e^(-z)   )
     * @param z
     * @return
     */
    static double sigma(double z){
        return 1/(1+Math.exp(-z));
    }

    default double applyRandomBias(int length, int numberOfNeuronsInPreviousLayer) {
        return this.getInitializationFunction().getRandomBias.apply(length, numberOfNeuronsInPreviousLayer);
    }


}

