package io.h3ca.forgemind.core.api;

import java.util.Iterator;
import java.util.NoSuchElementException;

public class DataLoader implements Iterator<Dataset.Batch> {

    private final Dataset dataset;
    private final int batchSize;
    private int currentIndex = 0;

    public DataLoader(Dataset dataset, int batchSize) {
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
    public Dataset.Batch next() {
        if (!hasNext()) throw new NoSuchElementException();

        Dataset.Batch batch = this.dataset.loadBatch(this.currentIndex, this.batchSize);
        this.currentIndex += this.batchSize;
        return batch;
    }

    public void reset() {
        this.currentIndex = 0;
    }

}
