package MLP.MNIST;

import MLP.Classification;
import MLP.Data.LabeledTrainingDataset;

import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class MnistTrainingDataset2 extends LabeledTrainingDataset<MnistImage, Classification> {

    public static final String IMAGES_PATH = "src/MLP/MNIST/data/train-images.idx3-ubyte";
    public static final String LABELS_PATH = "src/MLP/MNIST/data/train-labels.idx1-ubyte";
    public static final int SIZE = 60_000;
    public static final int OUTPUT_DIMENSION = 10;
    public static final int INPUT_DIMENSION = 784;

    private List<LabeledDataSample<MnistImage, Classification>> cachedData;
    private boolean isDataLoaded = false;

    public MnistTrainingDataset2(int batchSize) {
        super(SIZE, INPUT_DIMENSION, OUTPUT_DIMENSION, IMAGES_PATH, LABELS_PATH);
        this.batchSize = batchSize;
        this.cachedData = new ArrayList<>(SIZE);
    }

    /**
     * Loads the entire dataset into memory once
     */
    private void loadDataset() {
        if (isDataLoaded) return;

        try {
            // Read image file
            DataInputStream imageStream = new DataInputStream(
                    new BufferedInputStream(new FileInputStream(this.path)));

            // Read image header
            int imageMagic = imageStream.readInt();
            int numImages = imageStream.readInt();
            int numRows = imageStream.readInt();
            int numCols = imageStream.readInt();

            // Read label file
            DataInputStream labelStream = new DataInputStream(
                    new BufferedInputStream(new FileInputStream(this.labelPath)));

            // Read label header
            int labelMagic = labelStream.readInt();
            int numLabels = labelStream.readInt();

            if (numImages != numLabels) {
                throw new IOException("Image and label counts do not match");
            }

            // Limit to actual size
            int actualSize = Math.min(numImages, this.size);

            // Read data
            for (int i = 0; i < actualSize; i++) {
                // Read label
                int label = labelStream.readUnsignedByte();
                Classification classification = new Classification(this.outputDimension, label);

                // Read image
                MnistImage image = new MnistImage();
                for (int r = 0; r < numRows; r++) {
                    for (int c = 0; c < numCols; c++) {
                        image.getData()[r][c] = imageStream.readUnsignedByte();
                    }
                }

                this.cachedData.add(new LabeledDataSample<>(image, classification));
            }

            imageStream.close();
            labelStream.close();
            isDataLoaded = true;

        } catch (IOException e) {
            throw new RuntimeException("Failed to load MNIST dataset", e);
        }
    }

    @Override
    public double[] vectorizeInput(MnistImage input) {
        double[] result = new double[784]; // 28x28 = 784
        int idx = 0;

        // Flatten the 2D image into a 1D vector
        for (int r = 0; r < 28; r++) {
            for (int c = 0; c < 28; c++) {
                // Normalize pixel values to [0,1]
                result[idx++] = input.getData()[r][c] / 255.0;
            }
        }

        return result;
    }

    @Override
    public double[] vectorizeOutput(Classification input) {
        // Return one-hot encoded vector
        return input.getHotEncoding();
    }

    @Override
    public LabeledDataSample<MnistImage, Classification> load(int i) {
        if (!isDataLoaded) {
            loadDataset();
        }

        if (i < 0 || i >= this.cachedData.size()) {
            throw new IndexOutOfBoundsException("Index " + i + " is out of bounds for dataset size " + this.cachedData.size());
        }

        return this.cachedData.get(i);
    }

    @Override
    public List<LabeledDataSample<MnistImage, Classification>> loadList(int a, int b) {
        if (!isDataLoaded) {
            loadDataset();
        }

        if (a < 0 || b > this.cachedData.size() || a > b) {
            throw new IndexOutOfBoundsException("Invalid range [" + a + ", " + b + "] for dataset size " + this.cachedData.size());
        }

        return this.cachedData.subList(a, b);
    }
}