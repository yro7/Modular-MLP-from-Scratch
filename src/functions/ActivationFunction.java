package functions;

import matrices.ActivationMatrix;

import java.util.function.Function;

import static functions.InitializationFunction.*;

public enum ActivationFunction implements Function<ActivationMatrix,ActivationMatrix> {


    ReLU(He, activationMatrix -> activationMatrix.applyFunction(d -> Math.max(0, d)),
            activationMatrix -> activationMatrix.applyFunction(d -> d > 0.0 ? 1.0 : 0.0)
    ),

    TanH(Xavier, activationMatrix -> activationMatrix.applyFunction(Math::tanh),
            activationMatrix -> activationMatrix.applyFunction(d -> 1 - Math.pow(Math.tanh(d), 2))
    ),

    Sigmoid(LeCun, activationmatrix -> activationmatrix.applyFunction(d -> sigma(d)),
            activationMatrix -> activationMatrix.applyFunction(d -> sigma(d)*(1-sigma(d)))
    ),


    Identity(Xavier, activationMatrix -> activationMatrix,
            activationmatrix -> activationmatrix.applyFunction(d -> 1.0)),


    SoftMax(Xavier,
            activationmatrix -> {

                activationmatrix.centerOverRows();
                activationmatrix.applyFunction(Math::exp);
                double[] sumOverRows = activationmatrix.sumOverRows();
                activationmatrix.applyToElements((i, j) ->
                        activationmatrix.getData()[i][j] /= sumOverRows[i]);
                return activationmatrix;
            },

            activationMatrix -> activationMatrix

    );

    public final InitializationFunction initializationFunction;
    public final Function<ActivationMatrix, ActivationMatrix> function;
    public final Function<ActivationMatrix,ActivationMatrix> derivativeFunction;


    ActivationFunction(InitializationFunction initializationFunction, Function<ActivationMatrix, ActivationMatrix> function,
                        Function<ActivationMatrix, ActivationMatrix> derivativeFunction) {
        this.initializationFunction = initializationFunction;
        this.function = function;
        this.derivativeFunction = derivativeFunction;
    }

    @Override
    public ActivationMatrix apply(ActivationMatrix value) {
        return function.apply(value);
    }

    public InitializationFunction getInitializationFunction() {
        return initializationFunction;
    }

    public double applyRandomBias(int n, int p){
        return this.getInitializationFunction().getRandomBias.apply(n,p);
    }

    public Function<ActivationMatrix, ActivationMatrix> getDerivative() {
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
