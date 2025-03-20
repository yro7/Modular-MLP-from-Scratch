package Function;

import java.util.Random;
import java.util.function.Function;

import static Function.InitializationFunction.*;

public enum ActivationFunction implements Function<Double,Double> {


    ReLU(He, a -> Math.max(0, a),
            d -> d > 0.0 ? 1.0 : 0.0
    ),

    TanH(Xavier, Math::tanh,
            d -> 1 - Math.pow(Math.tanh(d), 2)
    ),

    Sigmoid(LeCun,
            d -> sigma(d),
            d -> sigma(d)*(1-sigma(d))
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

    /**
     * Renvoie la sigmoïde de z, càd
     * 1 / (    1 + e^(-z)   )
     * @param z
     * @return
     */
    public static final double sigma(double z){
        return 1/(1+Math.exp(-z));
    }
}

