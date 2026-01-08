package io.h3ca.forgemind.core.layers.activation;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.JsonTypeName;
import io.h3ca.forgemind.core.api.Tensor;
import io.h3ca.forgemind.core.layers.LayerBuilder;
import io.h3ca.forgemind.core.layers.Layer;
import tools.jackson.databind.annotation.JsonDeserialize;
import tools.jackson.databind.annotation.JsonPOJOBuilder;

@JsonDeserialize(builder = Softmax.Builder.class)
public class Softmax extends Layer {

    private final int outputSize;

    public Softmax(int outputSize) {
        super();
        if (outputSize <= 0) throw new IllegalArgumentException("outputSize must be positive");
        this.outputSize = outputSize;
    }

    @JsonTypeName("SoftmaxBuilder")
    @JsonPOJOBuilder(withPrefix = "")
    public static class Builder implements LayerBuilder {

        @JsonProperty("outputSize")
        private final int outputSize;

        @JsonCreator
        public Builder(@JsonProperty("outputSize") int outputSize) {
            super();
            if (outputSize <= 0) throw new IllegalArgumentException("outputSize must be positive");
            this.outputSize = outputSize;
        }

        @Override
        public Layer build() {
            return new Softmax(this.outputSize);
        }
    }

    @Override
    public Tensor forward(Tensor input) {
        this.batchSize = input.getShape()[0];

        this.cache.ensureOutput(new int[]{this.batchSize, this.outputSize});
        this.cache.ensureDeltaOutput(new int[]{this.batchSize, this.outputSize});

        this.cache.reset();

        Tensor features = this.cache.getOutput();
        float[] featuresData = features.getData();
        int[] featuresSizes = features.getStrides();
        int[] featuresStrides = features.getStrides();

        float[] inputData = input.getData();
        int[] inputStrides = input.getStrides();

        for (int sampleId = 0; sampleId < this.batchSize; sampleId++) {

            float max = Float.NEGATIVE_INFINITY;
            for (int featureId = 0; featureId < featuresSizes[0]; featureId++) {
                int inputOffset = sampleId * inputStrides[0] + featureId * inputStrides[1];
                max = Math.max(max, inputData[inputOffset]);
            }

            float sum = 0;
            for (int featureId = 0; featureId < featuresSizes[0]; featureId++) {
                int featuresOffset = sampleId * featuresStrides[0] + featureId * featuresStrides[1];
                float e = (float) Math.exp(inputData[featuresOffset] - max);

                featuresData[featuresOffset] = e;
                sum += e;
            }

            if (sum == 0) sum = 1e-12f;

            float invSum = (float) (1.0 / sum);
            for (int featureId = 0; featureId < featuresSizes[0]; featureId++) {
                int featuresOffset = sampleId * featuresStrides[0] + featureId * featuresStrides[1];
                featuresData[featuresOffset] *= invSum;
            }
        }

        return features;
    }

    @Override
    public Tensor backward(Tensor deltaOutput) {
        this.cache.setDeltaOutput(deltaOutput);
        return this.cache.getDeltaOutput();
    }
}
