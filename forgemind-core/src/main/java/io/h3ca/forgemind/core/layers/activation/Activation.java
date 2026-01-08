package io.h3ca.forgemind.core.layers.activation;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.JsonTypeName;
import io.h3ca.forgemind.core.api.Tensor;
import io.h3ca.forgemind.core.layers.LayerBuilder;
import io.h3ca.forgemind.core.layers.Layer;
import tools.jackson.databind.annotation.JsonDeserialize;
import tools.jackson.databind.annotation.JsonPOJOBuilder;

import java.util.Arrays;
import java.util.Objects;

@JsonDeserialize(builder = Activation.Builder.class)
public class Activation extends Layer {

    private final ActivationFunction activationFunction;

    public Activation(ActivationFunction activationFunction) {
        super();
        this.activationFunction = Objects.requireNonNull(activationFunction, "activationFunction cannot be null");
    }

    @JsonTypeName("ActivationBuilder")
    @JsonPOJOBuilder(withPrefix = "")
    public static class Builder implements LayerBuilder {

        @JsonProperty("activation")
        private final ActivationFunction activation;

        @JsonCreator
        public Builder(@JsonProperty("activation") ActivationFunction activation) {
            super();
            if (activation == null) throw new IllegalArgumentException("activation cannot be null.");
            this.activation = activation;
        }

        @Override
        public Layer build() {
            return new Activation(this.activation);
        }
    }

    @Override
    public Tensor forward(Tensor input) {
        this.batchSize = input.getShape()[0];

        this.cache.ensureOutput(input.getShape());
        this.cache.ensureDeltaOutput(input.getShape());

        this.cache.reset();

        Tensor features = this.cache.getOutput();
        float[] featuresData = features.getData();
        int[] featuresShape = features.getShape();
        int[] featuresStrides = features.getStrides();
        float[] inputData = input.getData();

        int numNeurons = Arrays.stream(featuresShape).skip(1).reduce(1, Math::multiplyExact);
        for (int sampleId = 0; sampleId < featuresShape[0]; sampleId++) {
            for (int neuronId = 0; neuronId < numNeurons; neuronId++) {
                int offset = sampleId * featuresStrides[0] + neuronId;
                featuresData[offset] = this.activationFunction.activate(inputData[offset]);
            }
        }

        return features;
    }

    @Override
    public Tensor backward(Tensor outputDelta) {
        this.cache.setDeltaOutput(outputDelta);

        Tensor features = this.cache.getOutput();
        int[] featureShape = features.getShape();
        float[] featuresData = features.getData();

        this.cache.ensureInputDelta(featureShape);
        Tensor inputDelta = this.cache.getInputDelta();
        this.cache.reset(inputDelta);

        float[] inputDeltaData = inputDelta.getData();
        int[] inputDeltaShape = inputDelta.getShape();
        int[] inputDeltaStrides = inputDelta.getStrides();

        float[] outputDeltaData = outputDelta.getData();

        for (int sampleId = 0; sampleId < inputDeltaShape[0]; sampleId++) {

            for (int neuronId = 0; neuronId < inputDeltaShape[1]; neuronId++) {
                int inputOffset = sampleId * inputDeltaStrides[0] + neuronId * inputDeltaStrides[1];

                inputDeltaData[inputOffset] += outputDeltaData[inputOffset]
                        * this.activationFunction.derivative(featuresData[inputOffset]);
            }
        }

        return inputDelta;
    }

    public static class Sigmoid extends Activation {
        public Sigmoid() {
            super(ActivationFunction.Sigmoid);
        }
    }

    public static class ReLu extends Activation {
        public ReLu() {
            super(ActivationFunction.ReLu);
        }
    }
}