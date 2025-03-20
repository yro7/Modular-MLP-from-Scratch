package MLP.MNIST;

import java.io.*;

public class MnistDataReader {

    public static MnistVector[] readData(String dataFilePath, String labelFilePath) throws IOException {

        DataInputStream dataInputStream = new DataInputStream(new BufferedInputStream(new FileInputStream(dataFilePath)));
        int magicNumber = dataInputStream.readInt();
        int numberOfItems = dataInputStream.readInt();
        int nRows = dataInputStream.readInt();
        int nCols = dataInputStream.readInt();

        System.out.println("magic number is " + magicNumber);
        System.out.println("number of items is " + numberOfItems);
        System.out.println("number of rows is: " + nRows);
        System.out.println("number of cols is: " + nCols);

        DataInputStream labelInputStream = new DataInputStream(new BufferedInputStream(new FileInputStream(labelFilePath)));
        int labelMagicNumber = labelInputStream.readInt();
        int numberOfLabels = labelInputStream.readInt();

        System.out.println("labels magic number is: " + labelMagicNumber);
        System.out.println("number of labels is: " + numberOfLabels);

        MnistVector[] data = new MnistVector[numberOfItems];

        assert numberOfItems == numberOfLabels;

        for(int i = 0; i < numberOfItems; i++) {
            MnistVector mnistVector = new MnistVector();
            mnistVector.setLabel(labelInputStream.readUnsignedByte());

            // Lire les données de l'image et les stocker dans le vecteur
            for (int r = 0; r < nRows; r++) {
                for (int c = 0; c < nCols; c++) {
                    // Convertir les coordonnées 2D (r,c) en index 1D
                    int idx = r * nCols + c;
                    mnistVector.setValue(idx, dataInputStream.readUnsignedByte());
                }
            }
            data[i] = mnistVector;
        }
        dataInputStream.close();
        labelInputStream.close();
        return data;
    }
}