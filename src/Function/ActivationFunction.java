package Function;

import java.util.function.Function;

public enum ActivationFunction {
    ReLU(InitializationFunction.He, a -> Math.max(0, a)),
    TanH(InitializationFunction.Xavier, Math::tanh),
    Sigmoid(InitializationFunction.LeCun, a -> 1 / (1 + Math.exp(-a))),
    SoftMax(InitializationFunction.LeCun, a -> 1 / (1 + Math.exp(-a)));  // SoftMax avec la même fonction pour l'instant.

    public final InitializationFunction initializationFunction;
    public final Function<Double, Double> function;

    ActivationFunction(InitializationFunction initializationFunction, Function<Double, Double> function) {
        this.initializationFunction = initializationFunction;
        this.function = function;
    }

    public double apply(double value) {
        return function.apply(value);
    }

    public InitializationFunction getInitializationFunction() {
        return initializationFunction;
    }
}

