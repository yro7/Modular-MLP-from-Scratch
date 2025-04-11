package Matrices;

public class Utils {

    public static int argmax(double[] array) {
        int maxIndex = 0;
        for (int i = 1; i < array.length; i++) {
            if (array[i] > array[maxIndex]) {
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    public static double[][] zeroArray(int numberOfRows, int numberOfColumns) {
        return new double[numberOfRows][numberOfColumns];
    }
}
