import java.util.Random;
import java.util.function.BiFunction;

public enum InitializationFunction {

    He(
            (inputSize, layerSize) -> new Random().nextGaussian() * Math.sqrt(2.0 / inputSize),
            (inputSize,layerSize) -> 0.01
    ),

    Xavier(
            (inputSize, layerSize) -> new Random().nextGaussian() * Math.sqrt(1.0 / (inputSize+layerSize)),
            (inputSize, layerSize) -> 0.01
    ),

    LeCun(
            (inputSize, layerSize) -> new Random().nextGaussian() * Math.sqrt(1.0 / inputSize),
            (inputSize, layerSize) -> 0.01
    );


    public final BiFunction<Integer,Integer,Double> getRandomWeight;
    public final BiFunction<Integer,Integer, Double> getRandomBias;

    InitializationFunction(BiFunction<Integer,Integer,Double> getRandmWeight, BiFunction<Integer,Integer,Double> getRandomBias){
        this.getRandomWeight = getRandmWeight;
        this.getRandomBias = getRandomBias;
    }

}
