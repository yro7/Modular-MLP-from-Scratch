package MLP.MNIST;

import MLP.Classification;
import MLP.Data.LabeledDataset;
import MLP.Data.LabeledTrainingDataset;

import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.concurrent.atomic.AtomicBoolean;

public class MnistTrainingDataset extends LabeledTrainingDataset<MnistImage, Classification> {

    public static final String imagesPath = "src/MLP/MNIST/data/t10k-images.idx3-ubyte";
    public static final String labelsPath = "src/MLP/MNIST/data/t10k-labels.idx1-ubyte";

    public MnistTrainingDataset(int batchSize, int size, int inputDimension, int outputDimension) {
        super(size, inputDimension, outputDimension, imagesPath, labelsPath);
        this.batchSize = batchSize;

    }

    @Override
    public double[] vectorizeInput(MnistImage input) {
        return new double[0];
    }

    @Override
    public double[] vectorizeOutput(Classification input) {
        return new double[0];
    }

    @Override
    public LabeledDataSample<MnistImage, Classification> load(int k) {
        try {
            DataInputStream dataInputStream = new DataInputStream(new BufferedInputStream(new FileInputStream(this.path)));
            int magicNumber = dataInputStream.readInt();
            int numberOfItems = dataInputStream.readInt();
            int nRows = dataInputStream.readInt();
            int nCols = dataInputStream.readInt();

            System.out.println("magic number is " + magicNumber);
            System.out.println("number of items is " + numberOfItems);
            System.out.println("number of rows is: " + nRows);
            System.out.println("number of cols is: " + nCols);

            DataInputStream labelInputStream = new DataInputStream(new BufferedInputStream(new FileInputStream(this.path)));
            int labelMagicNumber = labelInputStream.readInt();
            int numberOfLabels = labelInputStream.readInt();

            System.out.println("labels magic number is: " + labelMagicNumber);
            System.out.println("number of labels is: " + numberOfLabels);


            assert numberOfItems == numberOfLabels;

            MnistImage mnistImage = null;
            Classification label = null;

            for (int i = 0; i < numberOfItems; i++) {
                mnistImage = new MnistImage();

                int labelNumber = labelInputStream.readUnsignedByte();
                label = new Classification(this.outputDimension, 4); //TODO fix label import

                for (int r = 0; r < nRows; r++) {
                    for (int c = 0; c < nCols; c++) {
                        mnistImage.getData()[r][c] = dataInputStream.readUnsignedByte();
                    }
                }

            }
            dataInputStream.close();
            labelInputStream.close();

            return new LabeledDataSample<>(mnistImage, label);

        } catch(Exception e) { // TODO handle exceptions
            e.printStackTrace();
            throw new RuntimeException();
        }

    }

    private static int readInt(DataInputStream is) throws IOException {
        byte[] bytes = new byte[4];
        is.readFully(bytes);
        return ((bytes[0] & 0xFF) << 24) |
                ((bytes[1] & 0xFF) << 16) |
                ((bytes[2] & 0xFF) << 8) |
                (bytes[3] & 0xFF);


    }

}