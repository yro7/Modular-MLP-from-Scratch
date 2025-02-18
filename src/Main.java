public class Main {
    public static void main(String[] args) {
        System.out.println("Hello, World!");

        MLP mlpTest = MLP.builder(4)
                        .addLayer(1, ActivationFunction.ReLU)
                        .build();

        mlpTest.print();

        ActivationVector res = mlpTest.feedForward(ActivationVector.of(1.0, 2.0, 3.0, 4.0));

        System.out.println("Res : " + res);

    }


}