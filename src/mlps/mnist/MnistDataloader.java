package mlps.mnist;

import mlps.Classification;
import mlps.data.Loaders.Dataloader;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;

public class MnistDataloader extends Dataloader<MnistImage, Classification> {

    public static final int OUTPUT_DIMENSION = 10;
    public static final int INPUT_DIMENSION = 784;


    // Image file format constants
    private static final int IMAGE_HEADER_SIZE = 16; // 4 ints: magic, numImages, numRows, numCols
    private static final int LABEL_HEADER_SIZE = 8;  // 2 ints: magic, numLabels
    private static final int IMAGE_SIZE = 28 * 28;   // 28x28 pixels per image

    private int numRows = 28;
    private int numCols = 28;

    public MnistDataloader(String featuresPath, String labelPath, int batchSize, int size) {
        super(size, INPUT_DIMENSION, OUTPUT_DIMENSION, featuresPath, labelPath, batchSize);
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
        if (i < 0 || i >= this.size) {
            throw new IndexOutOfBoundsException("Index " + i + " is out of bounds for dataset size " + this.size);
        }

        return loadSingleImage(i);
    }


    /**
     * Loads a single image and its label from the MNIST dataset
     * // TODO fix ça pour load toute la batch à la fois plutot qu'1 image par 1 image (lent)
     */
    private LabeledDataSample<MnistImage, Classification> loadSingleImage(int index) {
        try {
            // Open files for reading

            FileInputStream imageFileStream = new FileInputStream(this.featuresPath);
            FileInputStream labelFileStream = new FileInputStream(this.labelPath);

            // Skip to the desired image
            long imageOffset = IMAGE_HEADER_SIZE + (long) index * IMAGE_SIZE;
            long actualImageSkip = imageFileStream.skip(imageOffset);
            if (actualImageSkip < imageOffset) {
                throw new IOException("Could not skip to the desired image position");
            }

            // Skip to the desired label
            long labelOffset = LABEL_HEADER_SIZE + index;
            long actualLabelSkip = 0;
            while (actualLabelSkip < labelOffset) {
                long skipped = labelFileStream.skip(labelOffset - actualLabelSkip);
                if (skipped == 0) {
                    throw new IOException("Could not skip to the desired label position");
                }
                actualLabelSkip += skipped;
            }

            // Read the label
            int label = labelFileStream.read();
            if (label < 0 || label >= this.outputDimension) {
                throw new IOException("Invalid label value: " + label);
            }
            Classification classification = new Classification(this.outputDimension, label);

            // Read the image
            MnistImage image = new MnistImage();
            byte[] imageBuffer = new byte[IMAGE_SIZE];
            imageFileStream.read(imageBuffer);

            // Convert to 2D array
            int idx = 0;
            for (int r = 0; r < numRows; r++) {
                for (int c = 0; c < numCols; c++) {
                    // Convert signed byte to unsigned (0-255)
                    image.getData()[r][c] = imageBuffer[idx++] & 0xFF;
                }
            }

            // Close streams
            imageFileStream.close();
            labelFileStream.close();

            return new LabeledDataSample<>(image, classification);

        } catch (IOException e) {
            throw new RuntimeException("Failed to load MNIST image at index " + index, e);
        }
    }

}
