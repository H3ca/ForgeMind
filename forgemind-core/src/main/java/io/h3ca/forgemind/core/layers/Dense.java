package io.h3ca.forgemind.core.layers;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.JsonTypeName;
import io.h3ca.forgemind.core.api.Tensor;
import io.h3ca.forgemind.core.api.Initializer;
import io.h3ca.forgemind.core.optimizers.Optimizer;
import tools.jackson.databind.annotation.JsonDeserialize;
import tools.jackson.databind.annotation.JsonPOJOBuilder;

import java.util.Objects;

@JsonTypeName("Dense")
@JsonDeserialize(builder = Dense.Builder.class)
public class Dense extends Layer {

    private static final int DENSE_BATCH = 0;
    private static final int DENSE_FEATURE = 1;

    private static final int WEIGHT_FEATURE_IN = 0;
    private static final int WEIGHT_FEATURE_OUT = 1;

    private final int outputSize;

    @JsonProperty("weights")
    private Tensor weights;
    @JsonProperty("biases")
    private Tensor biases;

    private final Initializer initializer;
    private Optimizer.StateInterface weightsOptimizerState;
    private Optimizer.StateInterface biasesOptimizerState;

    private Dense(int outputSize, Initializer initializer, Tensor weights, Tensor biases) {
        super();
        if (outputSize <= 0) throw new IllegalArgumentException("outputSize must be positive");
        this.initializer = Objects.requireNonNull(initializer, "initializer cannot be null");
        this.outputSize = outputSize;
        this.weights = weights;
        this.biases = biases;
    }

    @JsonTypeName("DenseBuilder")
    @JsonPOJOBuilder(withPrefix = "")
    public static class Builder implements LayerBuilder {

        @JsonProperty("outputSize")
        private final int outputSize;
        @JsonProperty("initializer")
        private Initializer initializer = new Initializer.GlorotUniform();
        @JsonProperty("weights")
        private Tensor weights;
        @JsonProperty("biases")
        private Tensor biases;

        @JsonCreator
        public Builder(@JsonProperty("outputSize") int outputSize) {
            super();
            if (outputSize <= 0) throw new IllegalArgumentException("outputSize must be positive");
            this.outputSize = outputSize;
        }

        public Builder initializer(Initializer initializer) {
            if (initializer == null) throw new IllegalArgumentException("initializer cannot be null");
            this.initializer = initializer;
            return this;
        }

        public Builder weights(Tensor weights) {
            this.weights = weights;
            return this;
        }

        public Builder biases(Tensor biases) {
            this.biases = biases;
            return this;
        }

        @Override
        public Layer build() {
            return new Dense(this.outputSize, this.initializer, this.weights, this.biases);
        }
    }

    public void initWeights(int channelsIn) {
        if (channelsIn <= 0) throw new IllegalArgumentException("channelsIn must be positive");
        this.weights = new Tensor(new int[]{channelsIn, this.outputSize});
        this.initializer.initialize(this.weights);

        this.biases = new Tensor(new int[]{this.outputSize});
    }

    private void initOptimizerState(Optimizer optimizer) {
        if (optimizer == null) throw new IllegalArgumentException("optimizer cannot be null.");

        this.weightsOptimizerState = optimizer.createState();
        this.biasesOptimizerState = optimizer.createState();

        this.weightsOptimizerState.initialize(this.weights.size());
        this.biasesOptimizerState.initialize(this.biases.size());
    }

    @Override
    public Tensor forward(Tensor input) {
        this.cache.setInput(input);

        float[] inputData = input.getData();
        int[] inputShape = input.getShape();
        int[] inputStrides = input.getStrides();

        if (this.weights == null) {
            this.initWeights(inputShape[DENSE_FEATURE]);
        }

        float[] weightsData = this.weights.getData();
        int[] weightShape = this.weights.getShape();
        int[] weightsStrides = this.weights.getStrides();
        float[] biasesData = this.biases.getData();

        this.cache.ensureWeightsGradients(weightShape);
        this.cache.ensureBiasesGradients(this.biases.getShape());
        this.cache.ensureOutput(new int[]{inputShape[DENSE_BATCH], weightShape[WEIGHT_FEATURE_OUT]});
        this.cache.reset();

        Tensor output = this.cache.getOutput();
        float[] outputData = output.getData();
        int[] outputStrides = output.getStrides();

        for (int sampleId = 0; sampleId < inputShape[DENSE_BATCH]; sampleId++) {
            for (int featureOut = 0; featureOut < weightShape[WEIGHT_FEATURE_OUT]; featureOut++) {
                int featureOffset = sampleId * outputStrides[DENSE_BATCH]
                        + featureOut * outputStrides[DENSE_FEATURE];

                outputData[featureOffset] = biasesData[featureOut];

                for (int featureIn = 0; featureIn < weightShape[WEIGHT_FEATURE_IN]; featureIn++) {
                    int inputOffset = sampleId * inputStrides[DENSE_BATCH]
                            + featureIn * inputStrides[DENSE_FEATURE];
                    int weightOffset = featureIn * weightsStrides[WEIGHT_FEATURE_IN]
                            + featureOut * weightsStrides[WEIGHT_FEATURE_OUT];

                    outputData[featureOffset] += inputData[inputOffset] * weightsData[weightOffset];
                }
            }
        }

        return output;
    }

    @Override
    public Tensor backward(Tensor outputDelta) {
        this.cache.setDeltaOutput(outputDelta);
        Tensor input = this.cache.getInput();

        this.cache.ensureInputDelta(input.getShape());

        Tensor deltaInput = this.cache.getInputDelta();
        this.cache.reset(deltaInput);

        float[] weightsData = this.weights.getData();
        int[] weightShape = this.weights.getShape();
        int[] weightsStrides = this.weights.getStrides();

        float[] deltaOutputData = outputDelta.getData();
        int[] deltaOutputStrides = outputDelta.getStrides();

        float[] deltaInputData = deltaInput.getData();
        int[] deltaInputShape = deltaInput.getShape();
        int[] deltaInputStrides = deltaInput.getStrides();

        for (int sampleId = 0; sampleId < deltaInputShape[DENSE_BATCH]; sampleId++) {
            for (int featureOut = 0; featureOut < weightShape[WEIGHT_FEATURE_OUT]; featureOut++) {
                int deltaInputOffset = sampleId * deltaOutputStrides[DENSE_BATCH]
                        + featureOut * deltaOutputStrides[DENSE_FEATURE];

                for (int featureIn = 0; featureIn < weightShape[WEIGHT_FEATURE_IN]; featureIn++) {
                    int deltaOutputOffset = sampleId * deltaInputStrides[DENSE_BATCH]
                            + featureIn * deltaInputStrides[DENSE_FEATURE];
                    int weightOffset = featureIn * weightsStrides[WEIGHT_FEATURE_IN]
                            +featureOut * weightsStrides[WEIGHT_FEATURE_OUT];
                    deltaInputData[deltaOutputOffset] += weightsData[weightOffset] * deltaOutputData[deltaInputOffset];
                }
            }
        }

        return deltaInput;
    }

    @Override
    public void update(Optimizer optimizer) {
        super.update(optimizer);

        if (this.weightsOptimizerState == null) this.initOptimizerState(optimizer);

        this.computeGradients();

        optimizer.apply(this.weightsOptimizerState, this.weights.getData(), this.cache.getWeightsGradient().getData());
        optimizer.apply(this.biasesOptimizerState, this.biases.getData(), this.cache.getBiasesGradient().getData());
    }

    @Override
    public void copyParametersFrom(Layer other) {
        if (!(other instanceof Dense source)) throw new IllegalArgumentException("Layer type mismatch.");

        this.weights = source.weights.clone();
        this.biases  = source.biases.clone();
    }

    private void computeGradients() {

        Tensor delta = this.cache.getDeltaOutput();
        float[] deltaData = delta.getData();
        int[] deltaShape = delta.getShape();
        int[] deltaStrides = delta.getStrides();

        Tensor input = this.cache.getInput();
        float[] inputData = input.getData();
        int[] inputStrides = input.getStrides();

        // Calculate batch gradients
        Tensor weightsGradient = this.cache.getWeightsGradient();
        float[] weightsGradientData = weightsGradient.getData();
        int[] weightsGradientShape = weightsGradient.getShape();
        int[] weightsGradientStrides = weightsGradient.getStrides();

        Tensor biasesGradient = this.cache.getBiasesGradient();
        float[] biasGradientData = biasesGradient.getData();
        this.cache.reset(weightsGradient);
        this.cache.reset(biasesGradient);

        for (int sampleId = 0; sampleId < deltaShape[DENSE_BATCH]; sampleId++) {

            for (int featureOut = 0; featureOut < weightsGradientShape[WEIGHT_FEATURE_OUT]; featureOut++) {
                int deltaOffset = sampleId * deltaStrides[DENSE_BATCH]
                        + featureOut * deltaStrides[DENSE_FEATURE];
                float deltaValue = deltaData[deltaOffset];

                biasGradientData[featureOut] += deltaValue;

                for (int featureIn = 0; featureIn < weightsGradientShape[WEIGHT_FEATURE_IN]; featureIn++) {
                    int inputOffset = sampleId * inputStrides[DENSE_BATCH]
                            + featureIn * inputStrides[DENSE_FEATURE];
                    int weightsGradientOffset = featureIn * weightsGradientStrides[WEIGHT_FEATURE_IN]
                            + featureOut * weightsGradientStrides[WEIGHT_FEATURE_OUT];
                    weightsGradientData[weightsGradientOffset] += inputData[inputOffset] * deltaValue;
                }
            }
        }

        // Average gradients over batch
        for (int i = 0; i < weightsGradientData.length; i++)
            weightsGradientData[i] /= deltaShape[DENSE_BATCH];
        for (int i = 0; i < biasGradientData.length; i++)
            biasGradientData[i] /= deltaShape[DENSE_BATCH];
    }
}
