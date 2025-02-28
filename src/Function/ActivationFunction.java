package Function;

import java.util.Random;
import java.util.function.Function;

public enum ActivationFunction implements Function<Double,Double> {


    ReLU(InitializationFunction.He, a -> Math.max(0, a),
            d -> (double) Integer.signum(d.compareTo(0.0))
            ),

    TanH(InitializationFunction.Xavier, Math::tanh,
            d -> 1 - Math.pow(Math.tan(d),2)
    ),

    Sigmoid(InitializationFunction.LeCun, a -> 1 / (1 + Math.exp(-a)),
            d -> (1/(1 + Math.exp(-d)) * (1 - 1/(1 + Math.exp(-d))))
            );

    /**SoftMax(InitializationFunction.LeCun,
            a -> 1 / (1 + Math.exp(-a)), d -> 0.0 );**/  // TODO SOFTMAX IMPLEMENTATION

    public static Random randomGenerator = new Random();


    public final InitializationFunction initializationFunction;
    public final Function<Double, Double> function;
    public final Function<Double,Double> derivativeFunction;


    ActivationFunction(InitializationFunction initializationFunction, Function<Double, Double> function, Function<Double, Double> derivativeFunction) {
        this.initializationFunction = initializationFunction;
        this.function = function;
        this.derivativeFunction = derivativeFunction;
    }

    @Override
    public Double apply(Double value) {
        return function.apply(value);
    }

    public InitializationFunction getInitializationFunction() {
        return initializationFunction;
    }

    public double applyRandomBias(int n, int p){
        return this.getInitializationFunction().getRandomBias.apply(n,p);
    }

}

