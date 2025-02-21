import Function.ActivationFunction;
import Matrices.ActivationMatrix;
import Matrices.WeightMatrix;

public class Main {
    public static void main(String[] args) {
        System.out.println("Hello, World!");

        ActivationVector activationVector = ActivationVector.of(1.0, 2.0);

        MLP mlp = MLP.builder(4)
                .addLayer(2, ActivationFunction.ReLU)
                .addLayer(4, ActivationFunction.Sigmoid)
                .addLayer(2, ActivationFunction.Sigmoid)
                .build();

        WeightMatrix matrix = new WeightMatrix(5,5, ActivationFunction.Sigmoid);
        ActivationMatrix am = new ActivationMatrix(5,5);

        am.multiplyByWeightMatrix(matrix);

        am.print();


    }


}