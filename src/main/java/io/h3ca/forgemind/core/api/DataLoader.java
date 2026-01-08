package io.h3ca.forgemind.core.api;

import io.h3ca.forgemind.datasets.MnistDataset;

import java.util.Iterator;
import java.util.NoSuchElementException;

public class DataLoader implements Iterator<DataLoader.Batch> {

    private final MnistDataset dataset;
    private final int batchSize;
    private int currentIndex = 0;

    public DataLoader(MnistDataset dataset, int batchSize) {
        if (dataset == null) throw new IllegalArgumentException("Dataset must not be null.");
        if (batchSize < 1) throw new IllegalArgumentException("Batch size must be > 0.");

        this.dataset = dataset;
        this.batchSize = batchSize;
    }

    @Override
    public boolean hasNext() {
        return this.currentIndex < this.dataset.getNumSamples();
    }

    @Override
    public Batch next() {
        if (!hasNext()) throw new NoSuchElementException();

        Batch batch = this.dataset.loadBatch(this.currentIndex, this.batchSize);
        this.currentIndex += this.batchSize;
        return batch;
    }

    public void reset() {
        this.currentIndex = 0;
    }

    public record Batch(Tensor x, Tensor y) {}

}
