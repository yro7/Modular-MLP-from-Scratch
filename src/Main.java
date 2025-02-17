public class Main {
    public static void main(String[] args) {
        System.out.println("Hello, World!");

        MLP mlpTest = MLP.builder(4)
                        .addLayer(2, ActivationFunction.ReLU)
                        .addLayer(4, ActivationFunction.TanH)
                        .addLayer(4, ActivationFunction.TanH)
                        .addLayer(2, ActivationFunction.ReLU)
                        .build();

        mlpTest.print();

        System.out.println("Goodbye, World!");

    }


}