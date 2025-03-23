package MLP.MNIST;

import java.io.*;

public class MnistDataReader {

    public static MnistVector[] readData(String dataFilePath, String labelFilePath) throws IOException {
        File dataFile = new File(dataFilePath);
        File labelFile = new File(labelFilePath);

        // Lecture du fichier d'images
        DataInputStream dataInputStream = new DataInputStream(new BufferedInputStream(new FileInputStream(dataFile)));

        // Lecture des entêtes (en format big-endian)
        int dataMagicNumber = readInt(dataInputStream);
        int numberOfItems = readInt(dataInputStream);
        int nRows = readInt(dataInputStream);
        int nCols = readInt(dataInputStream);

        int imageSize = nRows * nCols;


        // Lecture du fichier d'étiquettes
        DataInputStream labelInputStream = new DataInputStream(new BufferedInputStream(new FileInputStream(labelFile)));

        // Lecture des entêtes (en format big-endian)
        int labelMagicNumber = readInt(labelInputStream);
        int numberOfLabels = readInt(labelInputStream);


        // Création du tableau de vecteurs
        MnistVector[] data = new MnistVector[numberOfItems];

        byte[] byteBuffer = new byte[imageSize];

        // Lecture des données
        for (int i = 0; i < numberOfItems; i++) {
            // Création d'un nouveau vecteur
            MnistVector mnistVector = new MnistVector();

            // Lecture de l'étiquette
            mnistVector.setLabel(labelInputStream.readUnsignedByte());

            // Lecture de l'image complète dans un buffer
            int bytesRead = dataInputStream.read(byteBuffer);
            if (bytesRead != imageSize) {
                throw new IOException("Erreur lors de la lecture de l'image " + i + ": " +
                        bytesRead + " octets lus au lieu de " + imageSize);
            }

            // Conversion en tableau de doubles
            double[] values = new double[imageSize];
            for (int j = 0; j < imageSize; j++) {
                // Conversion de byte signé à valeur entre 0 et 1
                values[j] = (byteBuffer[j] & 0xFF) / 255.0;
            }

            mnistVector.setValues(values);
            data[i] = mnistVector;

            // Affichage de progression pour les grands ensembles
            if (i > 0 && i % 10000 == 0) {
                System.out.println(i + " vecteurs traités");
            }
        }

        dataInputStream.close();
        labelInputStream.close();
        return data;
    }

    // Lecture d'un entier en big-endian
    private static int readInt(DataInputStream is) throws IOException {
        byte[] bytes = new byte[4];
        is.readFully(bytes);
        return ((bytes[0] & 0xFF) << 24) |
                ((bytes[1] & 0xFF) << 16) |
                ((bytes[2] & 0xFF) << 8) |
                (bytes[3] & 0xFF);
    }
}