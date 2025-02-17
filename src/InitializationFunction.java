import java.util.Random;
import java.util.function.BiFunction;

public class InitializationFunction {

    public BiFunction<Integer,Integer,Double> getRandomWeight;
    public BiFunction<Integer,Integer,Double> getRandomBias;

    // Static access
    public static final InitializationFunction He = new He();
    public static final InitializationFunction Xavier = new Xavier();
    public static final InitializationFunction LeCun = new LeCun();

    public static final class He extends InitializationFunction {
        /**
         * On tire une paire poids/biais aléatoire selon l'initialization de He
         * Voir <a href="https://arxiv.org/abs/1502.01852">...</a>
         */
        public He() {
            this.getRandomWeight = (inputSize, layerSize) -> new Random().nextGaussian() * Math.sqrt(2.0 / inputSize);
            this.getRandomBias = (inputSize,layerSize) -> 0.01;
        }
    }

    public static final class Xavier extends InitializationFunction {
        /**
         * On tire une paire poids/biais aléatoire selon l'initialization de Xavier
         * Voir // TODO find xavier glorot initialization paper link on arxiv
         */
        public Xavier() {
            this.getRandomWeight = (inputSize,layerSize) -> new Random().nextGaussian() * Math.sqrt(1.0 / (inputSize+layerSize));
            this.getRandomBias = (inputSize,layerSize) -> 0.01;
        }
    }

    public static final class LeCun extends InitializationFunction {
        public LeCun() {
            this.getRandomWeight = (inputSize, layerSize) -> new Random().nextGaussian() * Math.sqrt(1.0 / inputSize);
            this.getRandomBias = (inputSize,layerSize) -> 0.01;
        }
    }

}
