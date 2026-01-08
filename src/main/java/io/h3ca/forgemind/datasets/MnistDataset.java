package io.h3ca.forgemind.datasets;

import io.h3ca.forgemind.core.api.DataLoader;
import io.h3ca.forgemind.core.api.Tensor;

import java.io.*;
import java.util.Objects;
import java.util.stream.IntStream;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;

public final class MnistDataset {

    private static final int IMAGE_MAGIC = 0x00000803;
    private static final int LABEL_MAGIC = 0x00000801;
    private static final int LABEL_HEADER_SIZE = 8;
    private static final int IMAGE_HEADER_SIZE = 16;

    private static final String MNIST_ZIP = "/datasets/mnist.zip";
    private static final String TRAIN_IMAGES = "train-images.idx3-ubyte";
    private static final String TRAIN_LABELS = "train-labels.idx1-ubyte";
    private static final String TEST_IMAGES = "t10k-images.idx3-ubyte";
    private static final String TEST_LABELS = "t10k-labels.idx1-ubyte";

    private final int numSamples;
    private final int rows;
    private final int cols;
    private final int channels = 1;
    private final int[] classes;

    private final byte[] allImages;
    private final byte[] allLabels;

    private float normalizationFactor = 1.0f;

    public MnistDataset(int numSamples, boolean train) {
        String imagesFile = train ? TRAIN_IMAGES : TEST_IMAGES;
        String labelsFile = train ? TRAIN_LABELS : TEST_LABELS;

        this.classes = IntStream.range(0, 10).toArray();

        try (InputStream in = getClass().getResourceAsStream(MNIST_ZIP);
             ZipInputStream zis = new ZipInputStream(new BufferedInputStream(Objects.requireNonNull(in)))) {
            byte[] imagesData = null;
            byte[] labelsData = null;

            ZipEntry entry;
            while ((entry = zis.getNextEntry()) != null) {
                if (entry.getName().equals(imagesFile)) {
                    imagesData = zis.readAllBytes();
                } else if (entry.getName().equals(labelsFile)) {
                    labelsData = zis.readAllBytes();
                }
                zis.closeEntry();
            }

            if (imagesData == null || labelsData == null) throw new IOException("MNIST files not found in the ZIP resource.");

            try (DataInputStream images = new DataInputStream(new ByteArrayInputStream(imagesData));
                 DataInputStream labels = new DataInputStream(new ByteArrayInputStream(labelsData))) {

                int imageMagic = images.readInt();
                if (imageMagic != IMAGE_MAGIC) throw new IOException("Invalid MNIST image file.");

                int labelMagic = labels.readInt();
                if (labelMagic != LABEL_MAGIC) throw new IOException("Invalid MNIST label file.");

                int totalImages = images.readInt();
                this.rows = images.readInt();
                this.cols = images.readInt();
                int totalLabels = labels.readInt();

                if (totalImages != totalLabels)
                    throw new IOException("MNIST image/label count mismatch.");

                this.numSamples = numSamples > 0 ? Math.min(totalImages, numSamples) : totalImages;
            }

            this.allImages = imagesData;
            this.allLabels = labelsData;

        } catch (IOException e) {
            throw new RuntimeException("Failed to initialize MNIST dataset from InputStream", e);
        }
    }

    public MnistDataset(boolean train) {
        this(0, train);
    }

    public DataLoader.Batch loadBatch(int index, int size) {
        if (index < 0 || index >= this.numSamples)
            throw new IllegalArgumentException("Index out of bounds: " + index + ".");
        if (size <= 0)
            throw new IllegalArgumentException("Size must be positive.");

        int numBatchSamples = Math.min(size, this.numSamples - index);
        int imageSize = this.rows * this.cols;

        float[] imageTensorData = new float[numBatchSamples * imageSize];
        float[] labelTensorData = new float[numBatchSamples * this.classes.length];

        for (int i = 0; i < numBatchSamples; i++) {
            // Labels: skip header (8 bytes)
            int labelIndex = LABEL_HEADER_SIZE + index + i;
            int label = this.allLabels[labelIndex] & 0xFF;
            labelTensorData[i * this.classes.length + label] = 1.0f;

            // Images: skip header (16 bytes)
            int imageOffset = IMAGE_HEADER_SIZE + (index + i) * imageSize;
            int tensorOffset = i * imageSize;
            for (int p = 0; p < imageSize; p++) {
                imageTensorData[tensorOffset + p] = (this.allImages[imageOffset + p] & 0xFF) / this.normalizationFactor;
            }
        }

        Tensor x = new Tensor(imageTensorData, new int[]{numBatchSamples, this.rows, this.cols, this.channels});
        Tensor y = new Tensor(labelTensorData, new int[]{numBatchSamples, this.classes.length});

        return new DataLoader.Batch(x, y);
    }

    public void setNormalizationFactor(float value) {
        if (value <= 0) throw new IllegalArgumentException("Normalization value must be positive.");
        this.normalizationFactor = value;
    }

    public int getNumSamples() {
        return this.numSamples;
    }

    public int[] getImageShape() {
        return new int[]{this.numSamples, this.rows, this.cols, this.channels};
    }

    public int[] getLabelShape() {
        return new int[]{this.numSamples, this.classes.length};
    }

    public int[] getClasses() {
        return this.classes.clone();
    }

    public Tensor getSample(int index) {
        return loadBatch(index, 1).x();
    }

    public Tensor getLabel(int index) {
        return loadBatch(index, 1).y();
    }
}
