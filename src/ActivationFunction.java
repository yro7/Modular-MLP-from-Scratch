import java.util.Random;
import java.util.function.BiFunction;
import java.util.function.Function;

public abstract class ActivationFunction implements Function<Double,Double> {

    /**
     * En function de la fonction d'activation choisie, on va initializer les
     * neurones différemment. Cela permet d'éviter des problèmes comme le
     * vanishing gradient problem pour les réseaux plus compliqués.
     */
    public InitializationFunction initializationFunction;


    /**
     * static Activation functions.
     */
    public static final ReLU ReLU = new ReLU();
    public static final TanH TanH = new TanH();
    public static final Sigmoid Sigmoid = new Sigmoid();

    /**
     * Initialization functions.
     */

    public static final InitializationFunction test = null;

    public static final class ReLU extends ActivationFunction {

        /**
         * On tire une paire poids/biais aléatoire selon l'initialization de He
         * Voir <a href="https://arxiv.org/abs/1502.01852">...</a>
         */
        public ReLU() {
            this.initializationFunction = InitializationFunction.He;
        }

        @Override
        public Double apply(Double aDouble) {
            return Math.max(0,aDouble);
        }
    }

    public static final class TanH extends ActivationFunction {

        /**
         * On tire une paire poids/biais aléatoire selon l'initialization Normale de Xavier Glorot
         * Voir TODO find arxiv link
         */
        public TanH() {
            this.initializationFunction = InitializationFunction.Xavier;
        }

        @Override
        public Double apply(Double aDouble) {
            return Math.tanh(aDouble);
        }
    }

    public static final class Sigmoid extends ActivationFunction {

        /**
         * On tire une paire poids/biais aléatoire selon l'initialization UNIFORME de Xavier Glorot
         * Voir TODO find arxiv link
         */
        public Sigmoid() {
            this.initializationFunction = InitializationFunction.LeCun;
        }

        @Override
        public Double apply(Double aDouble) {
            return 1 / (1 + Math.exp(-aDouble));

        }
    }

    public static final class SoftMax extends ActivationFunction {

        /**
         * On tire une paire poids/biais aléatoire selon l'initialization UNIFORME de Xavier Glorot
         * Voir TODO find arxiv link
         */
        public SoftMax() {
            this.initializationFunction = InitializationFunction.LeCun;
        }

        @Override
        public Double apply(Double aDouble) {
            return 1 / (1 + Math.exp(-aDouble));

        }
    }

}



