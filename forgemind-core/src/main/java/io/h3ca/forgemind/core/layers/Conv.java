package io.h3ca.forgemind.core.layers;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.JsonTypeName;
import io.h3ca.forgemind.core.api.Tensor;
import io.h3ca.forgemind.core.api.Initializer;
import io.h3ca.forgemind.core.internal.ConvNative;
import io.h3ca.forgemind.core.optimizers.Optimizer;
import io.h3ca.forgemind.core.utils.NativeMemoryUtils;
import tools.jackson.databind.annotation.JsonDeserialize;
import tools.jackson.databind.annotation.JsonPOJOBuilder;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.Arrays;
import java.util.Objects;

@JsonTypeName("Conv")
@JsonDeserialize(builder = Conv.Builder.class)
public class Conv extends Layer {
    // Assumes: stride = 1, SAME padding, odd kernel size
    private static final int NHWC_BATCH = 0;
    private static final int NHWC_HEIGHT = 1;
    private static final int NHWC_WIDTH = 2;
    private static final int NHWC_CHANNEL = 3;

    private static final int KERNEL_HEIGHT = 0;
    private static final int KERNEL_WIDTH = 1;
    private static final int KERNEL_CHANNEL_IN = 2;
    private static final int KERNEL_CHANNEL_OUT = 3;

    private static final int PADDING_WIDTH = 0;
    private static final int PADDING_HEIGHT = 1;

    @JsonProperty("weights")
    private Tensor kernel;
    @JsonProperty("biases")
    private Tensor biases;
    private final Initializer initializer;
    private Optimizer.StateInterface kernelOptimizerState;
    private Optimizer.StateInterface biasesOptimizerState;

    // [WIDTH, HEIGHT]
    private final int[] padding = new int[2];
    private final int numFilters;
    private final int[] kernelShape;

    // Buffers
    private float[] patchCache;

    /**
     * Creates a convolution layer.
     *
     * @param kernelShape kernel height and width
     * @throws IllegalArgumentException if kernelShape does not have length 2
     *                                  or if kernel dimensions are even
     */
    private Conv(int numFilters, int[] kernelShape, Initializer initializer, Tensor kernel, Tensor biases) {
        super();

        if (numFilters <= 0) throw new IllegalArgumentException("numFilters must be positive.");
        validateKernel(kernelShape);

        this.initializer = Objects.requireNonNull(initializer, "initializer cannot be null");

        this.numFilters = numFilters;
        this.kernelShape = kernelShape.clone();
        this.padding[PADDING_WIDTH] = Math.floorDiv(this.kernelShape[KERNEL_HEIGHT], 2);
        this.padding[PADDING_HEIGHT] = Math.floorDiv(this.kernelShape[KERNEL_WIDTH], 2);
        this.kernel = kernel;
        this.biases = biases;
    }

    @JsonTypeName("ConvBuilder")
    @JsonPOJOBuilder(withPrefix = "")
    public static class Builder implements LayerBuilder {

        @JsonProperty("numFilters")
        private final int numFilters;
        @JsonProperty("kernelShape")
        private final int[] kernelShape;
        @JsonProperty("initializer")
        private Initializer initializer = new Initializer.GlorotUniform();
        @JsonProperty("weights")
        private Tensor weights;
        @JsonProperty("biases")
        private Tensor biases;

        public Builder(@JsonProperty("numFilters") int numFilters, @JsonProperty("kernelShape") int[] kernelShape) {
            if (numFilters <= 0) throw new IllegalArgumentException("numFilters must be positive.");
            validateKernel(kernelShape);

            this.numFilters = numFilters;
            this.kernelShape = kernelShape.clone();
        }

        public Builder initializer(Initializer initializer) {
            if (initializer == null) throw new IllegalArgumentException("initializer cannot be null.");
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
            if (this.weights != null && this.biases == null)
                throw new IllegalArgumentException("Biases must be provided if weights are set.");

            return new Conv(this.numFilters, this.kernelShape, this.initializer, this.weights, this.biases);
        }
    }

    private static void validateKernel(int[] kernelShape) {
        if (kernelShape == null) throw new IllegalArgumentException("kernelShape cannot be null.");
        if (kernelShape.length != 2) throw new IllegalArgumentException(
                "Conv kernelShape must have length 2 [height, width], got " + kernelShape.length + ".");
        if (kernelShape[0] <= 0) throw new IllegalArgumentException("kernel height must be positive.");
        if (kernelShape[1] <= 0) throw new IllegalArgumentException("kernel width must be positive.");
        if (kernelShape[0] % 2 == 0 || kernelShape[1] % 2 == 0)
            throw new IllegalArgumentException("Kernel size must be odd for SAME padding.");
    }

    private void initWeights(int channelsIn) {
        if (channelsIn <= 0) throw new IllegalArgumentException("channelsIn must be positive");

        this.kernel = new Tensor(new int[]{this.kernelShape[0], this.kernelShape[1], channelsIn, this.numFilters});
        this.initializer.initialize(this.kernel);

        this.biases = new Tensor(new int[]{this.numFilters});
    }

    private void initOptimizerState(Optimizer optimizer) {
        if (optimizer == null) throw new IllegalArgumentException("optimizer cannot be null.");

        this.kernelOptimizerState = optimizer.createState();
        this.biasesOptimizerState = optimizer.createState();

        this.kernelOptimizerState.initialize(this.kernel.size());
        this.biasesOptimizerState.initialize(this.biases.size());
    }

    @Override
    public Tensor forward(Tensor input) {
        // Input Data
        this.cache.setInput(input);
        float[] inputData = input.getData();
        int[] inputShape = input.getShape();
        int[] inputStrides = input.getStrides();

        if (input.rank() != 4) throw new IllegalArgumentException("Conv expects 4D tensors.");

        if (this.kernel == null) this.initWeights(inputShape[3]);

        // Kernel Data
        float[] kernelData = this.kernel.getData();
        int[] kernelShape = this.kernel.getShape();

        // Bias
        float[] biasesData = this.biases.getData();

        if (kernelShape[2] != inputShape[3]) {
            throw new IllegalArgumentException("Kernel channels must match input channels\n" +
                    "Kernel shape: " + Arrays.toString(kernelShape) + "\n"+
                    "Input shape: " + Arrays.toString(inputShape));
        }

        // Ensure gradients
        this.cache.ensureWeightsGradients(kernelShape);
        this.cache.ensureBiasesGradients(this.biases.getShape());
        // Output Data
        this.cache.ensureOutput(new int[]{
                inputShape[NHWC_BATCH],
                inputShape[NHWC_HEIGHT],
                inputShape[NHWC_WIDTH],
                kernelShape[KERNEL_CHANNEL_OUT]
        });
        this.cache.reset();
        Tensor output = this.cache.getOutput();
        float[] outputData = output.getData();
        int[] outputStride = output.getStrides();

        int patchSize = kernelShape[KERNEL_HEIGHT]
                * kernelShape[KERNEL_WIDTH]
                * kernelShape[KERNEL_CHANNEL_IN];
        if (this.patchCache == null) {
            this.patchCache = new float[patchSize];
        }

        for (int sampleId = 0; sampleId < inputShape[NHWC_BATCH]; sampleId++) {
            for (int rowId = 0; rowId < inputShape[NHWC_HEIGHT]; rowId++) {
                for (int colId = 0; colId < inputShape[NHWC_WIDTH]; colId++) {
                    // Extract patch
                    int patchIndex = 0;
                    for (int kernelRowId = 0; kernelRowId < kernelShape[KERNEL_HEIGHT]; kernelRowId++) {
                        int inputRow = rowId + kernelRowId - this.padding[PADDING_HEIGHT];
                        for (int kernelColId = 0; kernelColId < kernelShape[KERNEL_WIDTH]; kernelColId++) {
                            int inputCol = colId + kernelColId - this.padding[PADDING_WIDTH];
                            for (int channelIn = 0; channelIn < kernelShape[KERNEL_CHANNEL_IN]; channelIn++) {
                                if (inputRow < 0 || inputRow >= inputShape[NHWC_HEIGHT]
                                        || inputCol < 0 || inputCol >= inputShape[NHWC_WIDTH]) {
                                    this.patchCache[patchIndex++] = 0.0f;
                                } else {
                                    int inputOffset = sampleId * inputStrides[NHWC_BATCH]
                                            + inputRow * inputStrides[NHWC_HEIGHT]
                                            + inputCol * inputStrides[NHWC_WIDTH]
                                            + channelIn * inputStrides[NHWC_CHANNEL];
                                    this.patchCache[patchIndex++] = inputData[inputOffset];
                                }
                            }
                        }
                    }

                    // Perform convolution
                    for (int channelOut = 0; channelOut < kernelShape[KERNEL_CHANNEL_OUT]; channelOut++) {
                        float sum = biasesData[channelOut];

                        int kernelIndex = channelOut;
                        for (int k = 0; k < patchSize; k++) {
                            sum += this.patchCache[k] * kernelData[kernelIndex];
                            kernelIndex += kernelShape[KERNEL_CHANNEL_OUT];
                        }

                        int outputOffset = sampleId * outputStride[NHWC_BATCH]
                                + rowId * outputStride[NHWC_HEIGHT]
                                + colId * outputStride[NHWC_WIDTH]
                                + channelOut * outputStride[NHWC_CHANNEL];

                        outputData[outputOffset] = sum;
                    }
                }
            }
        }

        return output;
    }

    @Override
    public Tensor backward(Tensor deltaOutput) {
        this.cache.setDeltaOutput(deltaOutput);

        this.cache.ensureInputDelta(this.cache.getInput().getShape());
        Tensor deltaInput = this.cache.getInputDelta();
        this.cache.reset(deltaInput);

        float[] deltaOutputData = deltaOutput.getData();
        int[] deltaOutputShape = deltaOutput.getShape();
        int[] deltaOutputStrides = deltaOutput.getStrides();

        float[] deltaInputData = deltaInput.getData();
        int[] deltaInputShape = deltaInput.getShape();
        int[] deltaInputStrides = deltaInput.getStrides();

        float[] kernelData = this.kernel.getData();
        int[] kernelShape = this.kernel.getShape();
        int[] kernelStrides = this.kernel.getStrides();

        for (int sampleId = 0; sampleId < deltaInputShape[NHWC_BATCH]; sampleId++) {
            for (int channelIn = 0; channelIn < kernelShape[KERNEL_CHANNEL_IN]; channelIn++) {
                for (int rowId = 0; rowId < deltaInputShape[NHWC_HEIGHT]; rowId++) {
                    for (int colId = 0; colId < deltaInputShape[NHWC_WIDTH]; colId++) {

                        float grad = 0.0f;
                        for (int kernelRowId = 0; kernelRowId < kernelShape[KERNEL_HEIGHT]; kernelRowId++) {
                            int inputRow = rowId + this.padding[PADDING_HEIGHT] - kernelRowId;
                            if (inputRow < 0 || inputRow >= deltaOutputShape[NHWC_HEIGHT]) continue;

                            for (int kernelColId = 0; kernelColId < kernelShape[KERNEL_WIDTH]; kernelColId++) {
                                int inputCol = colId + this.padding[PADDING_WIDTH] - kernelColId;
                                if (inputCol < 0 || inputCol >= deltaOutputShape[NHWC_WIDTH]) continue;

                                for (int channelOut = 0; channelOut < kernelShape[KERNEL_CHANNEL_OUT]; channelOut++) {
                                    int deltaOutputOffset = sampleId * deltaOutputStrides[NHWC_BATCH]
                                            + inputRow * deltaOutputStrides[NHWC_HEIGHT]
                                            + inputCol * deltaOutputStrides[NHWC_WIDTH]
                                            + channelOut * deltaOutputStrides[NHWC_CHANNEL];
                                    int kernelOffset = kernelRowId * kernelStrides[KERNEL_HEIGHT]
                                            + kernelColId * kernelStrides[KERNEL_WIDTH]
                                            + channelIn * kernelStrides[KERNEL_CHANNEL_IN]
                                            + channelOut * kernelStrides[KERNEL_CHANNEL_OUT];

                                    grad += deltaOutputData[deltaOutputOffset] * kernelData[kernelOffset];

                                }
                            }
                        }
                        int deltaInputOffset = sampleId * deltaInputStrides[NHWC_BATCH]
                                + rowId * deltaInputStrides[NHWC_HEIGHT]
                                + colId * deltaInputStrides[NHWC_WIDTH]
                                + channelIn * deltaInputStrides[NHWC_CHANNEL];
                        deltaInputData[deltaInputOffset] = grad;
                    }
                }
            }
        }

        return deltaInput;
    }

    @Override
    public void update(Optimizer optimizer) {
        super.update(optimizer);

        if (this.kernelOptimizerState == null) this.initOptimizerState(optimizer);

        this.computeGradients();
        optimizer.apply(this.kernelOptimizerState, this.kernel.getData(), this.cache.getWeightsGradient().getData());
        optimizer.apply(this.biasesOptimizerState, this.biases.getData(), this.cache.getBiasesGradient().getData());
    }

    @Override
    public void copyParametersFrom(Layer other) {
        if (!(other instanceof Conv source)) throw new IllegalArgumentException("Layer type mismatch.");

        this.kernel = source.kernel.clone();
        this.biases  = source.biases.clone();
    }

    private void computeGradients() {
        Tensor deltaOutput = this.cache.getDeltaOutput();
        Tensor weightsGradient = this.cache.getWeightsGradient();
        Tensor biasesGradient = this.cache.getBiasesGradient();

        this.cache.reset(weightsGradient);
        this.cache.reset(biasesGradient);

        try (Arena arena = Arena.ofConfined()) {

            MemorySegment nativePadding = NativeMemoryUtils.fromIntArray(arena, this.padding);

            MemorySegment nativeDelta = deltaOutput.toNative(arena);
            MemorySegment nativeInput = this.cache.getInput().toNative(arena);
            MemorySegment nativeBiasesGradient = biasesGradient.toNative(arena);
            MemorySegment nativeWeightsGradient = weightsGradient.toNative(arena);

            ConvNative.computeGradients(nativePadding, nativeDelta, nativeInput, nativeBiasesGradient, nativeWeightsGradient);

            biasesGradient.fromNative(nativeBiasesGradient);
            weightsGradient.fromNative(nativeWeightsGradient);
        }
    }
}
