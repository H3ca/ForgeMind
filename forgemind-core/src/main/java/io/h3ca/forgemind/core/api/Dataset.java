package io.h3ca.forgemind.core.api;

public interface Dataset {

    int getNumSamples();
    Batch loadBatch(int index, int size);

    record Batch(Tensor x, Tensor y) {}
}
