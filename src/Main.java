import Function.ActivationFunction;
import Matrices.Matrix;

public class Main {
    public static void main(String[] args) {
        System.out.println("Hello, World!");

        MLP mlp = MLP.builder(4)
                .addLayer(2, ActivationFunction.ReLU)
                .addLayer(4, ActivationFunction.Sigmoid)
                .addLayer(2, ActivationFunction.Sigmoid)
                .build();


        Matrix matrix = new Matrix(5,5);
        matrix.print();

    }


}