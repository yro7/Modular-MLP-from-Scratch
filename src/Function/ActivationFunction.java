package Function;

import java.util.Random;
import java.util.function.Function;

import static Function.InitializationFunction.*;

public enum ActivationFunction implements Function<Double,Double> {


    ReLU(He, a -> Math.max(0, a),
            d -> (double) Integer.signum(d.compareTo(0.0))
    ),

    TanH(Xavier, Math::tanh,
            d -> 1 - Math.pow(Math.tan(d),2)
    ),

    Sigmoid(LeCun,
            a -> 1 / (1 + Math.exp(-a)),
            d -> (1/(1 + Math.exp(-d)) * (1 - 1/(1 + Math.exp(-d))))
            ),


    Identity(Xavier, d -> d, d -> 1.0);

     // TODO SOFTMAX IMPLEMENTATION : change implements double double to implements matrix double.

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

    public Function<Double, Double> getDerivative() {
        return this.derivativeFunction;
    }
}

