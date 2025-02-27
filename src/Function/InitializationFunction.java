package Function;

import java.util.Random;
import java.util.function.BiFunction;

import static Function.ActivationFunction.randomGenerator;

/**
 * En fonction de quelle {@link ActivationFunction} on choisit, il est utile d'initialiser les
 * poids et biais du réseau en fonction de certaines distributions aléatoires.
 * Les {@link InitializationFunction} permettent d'optimiser l'initialisation.
 */
public enum InitializationFunction {


    /**
     * Voir <a href="https://arxiv.org/pdf/1502.01852">Delving Deep into Rectifiers:
     * Surpassing Human-Level Performance on ImageNet Classification</a>.
     */
    He(
            (inputSize, layerSize) -> randomGenerator.nextGaussian() * Math.sqrt(2.0 / inputSize),
            (inputSize,layerSize) -> 0.01
    ),

    /**
     * Voir <a href="http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf">249Understanding the difficulty of training deep feedforward neural networks</a>.
     */
    Xavier(
            (inputSize, layerSize) -> randomGenerator.nextGaussian() * Math.sqrt(1.0 / (inputSize+layerSize)),
            (inputSize, layerSize) -> 0.01
    ),

    LeCun(
            (inputSize, layerSize) -> randomGenerator.nextGaussian() * Math.sqrt(1.0 / inputSize),
            (inputSize, layerSize) -> 0.01
    );


    public final BiFunction<Integer,Integer,Double> getRandomWeight;
    public final BiFunction<Integer,Integer, Double> getRandomBias;

    InitializationFunction(BiFunction<Integer,Integer,Double> getRandmWeight, BiFunction<Integer,Integer,Double> getRandomBias){
        this.getRandomWeight = getRandmWeight;
        this.getRandomBias = getRandomBias;
    }

}
