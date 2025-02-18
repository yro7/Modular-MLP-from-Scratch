public class Main {
    public static void main(String[] args) {
        System.out.println("Hello, World!");

        MLP mlpTest = MLP.builder(4)
                        .addLayer(1, ActivationFunction.ReLU)
                        .build();

        mlpTest.print();
        ActivationVector testVector4 = ActivationVector.of(1.0, 2.0, 3.0, 4.0);
        ActivationVector testVector1 = ActivationVector.of(1.0);

        double loss = mlpTest.computeLoss(testVector4, testVector1);
        System.out.println("Loss : " + loss);

    }


}