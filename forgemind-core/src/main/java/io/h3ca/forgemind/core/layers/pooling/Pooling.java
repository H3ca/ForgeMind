package io.h3ca.forgemind.core.layers.pooling;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.JsonTypeName;
import io.h3ca.forgemind.core.api.Tensor;
import io.h3ca.forgemind.core.layers.LayerBuilder;
import io.h3ca.forgemind.core.layers.Layer;
import io.h3ca.forgemind.core.layers.LayerCache;
import tools.jackson.databind.annotation.JsonDeserialize;
import tools.jackson.databind.annotation.JsonPOJOBuilder;

@JsonDeserialize(builder = Pooling.Builder.class)
public abstract class Pooling extends Layer {

    protected final int width, height;
    protected final int padWidth, padHeight;
    protected Tensor input;

    protected Pooling(int width, int height, int padWidth, int padHeight) {
        if (width <= 0) throw new IllegalArgumentException("width must be positive");
        if (height <= 0) throw new IllegalArgumentException("height must be positive");
        if (padWidth < 0) throw new IllegalArgumentException("padWidth cannot be negative");
        if (padHeight < 0) throw new IllegalArgumentException("padHeight cannot be negative");

        this.cache = new LayerCache();

        this.width = width;
        this.height = height;
        this.padWidth = padWidth;
        this.padHeight = padHeight;
    }

    @JsonTypeName("PoolingBuilder")
    @JsonPOJOBuilder(withPrefix = "")
    public static class Builder implements LayerBuilder {

        public enum Type { AVG, MAX, MIN, SUM }

        @JsonProperty("width")
        private final int width;
        @JsonProperty("height")
        private final int height;
        @JsonProperty("padWidth")
        private int padWidth = 0;
        @JsonProperty("padHeight")
        private int padHeight = 0;
        @JsonProperty("poolingType")
        private final Builder.Type poolingType;

        @JsonCreator
        public Builder(
                @JsonProperty("width") int width,
                @JsonProperty("height") int height,
                @JsonProperty("poolingType") Builder.Type poolingType
        ) {
            super();
            if (width <= 0) throw new IllegalArgumentException("width must be positive");
            if (height <= 0) throw new IllegalArgumentException("height must be positive");
            if (poolingType == null) throw new IllegalArgumentException("poolingType cannot be null");
            this.width = width;
            this.height = height;
            this.poolingType = poolingType;
        }

        public Builder(int width, int height) {
            this(width, height, Builder.Type.AVG);
        }

        public Builder padWidth(int padWidth) {
            if (padWidth < 0) throw new IllegalArgumentException("padWidth cannot be negative");
            this.padWidth = padWidth;
            return this;
        }

        public Builder padHeight(int padHeight) {
            if (padHeight < 0) throw new IllegalArgumentException("padHeight cannot be negative");
            this.padHeight = padHeight;
            return this;
        }

        @Override
        public Layer build() {
            return switch (this.poolingType) {
                case AVG -> new AvgPooling(this.width, this.height, this.padWidth, this.padHeight);
                case MAX -> new MaxPooling(this.width, this.height, this.padWidth, this.padHeight);
                case MIN -> new MinPooling(this.width, this.height, this.padWidth, this.padHeight);
                case SUM -> new SumPooling(this.width, this.height, this.padWidth, this.padHeight);
            };
        }
    }

    protected abstract float poolingFunction(float a, float b);
    protected abstract float getIdentityValue();
    protected abstract void distributeGradient(float[] deltaInput, float[] inputData,
                                               int sampleId, int channelId,
                                               int startY, int startX,
                                               int poolHeight, int poolWidth,
                                               int padHeight, int padWidth,
                                               float grad);

    @Override
    public Tensor forward(Tensor input) {
        this.input = input;
        int[] inputShape = input.getShape();
        this.batchSize = inputShape[0];

        int outputHeight = (inputShape[1] + 2 * this.padHeight - this.height) / this.height + 1;
        int outputWidth = (inputShape[2] + 2 * this.padWidth - this.width) / this.width + 1;

        this.cache.ensureOutput(new int[]{batchSize, outputHeight, outputWidth, inputShape[3]});
        float[] outData = this.cache.getOutput().getData();

        this.iterateForward(input, outData, inputShape[3], outputHeight, outputWidth);

        return this.cache.getOutput();
    }

    @Override
    public Tensor backward(Tensor deltaOutput) {
        int outputHeight = deltaOutput.getShape()[1];
        int outputWidth = deltaOutput.getShape()[2];
        int numChannels = deltaOutput.getShape()[3];

        Tensor deltaInput = new Tensor(this.input.getShape());
        float[] deltaData = deltaOutput.getData();
        float[] deltaInputData = deltaInput.getData();

        iterateBackward(deltaData, deltaInputData, numChannels, outputHeight, outputWidth);

        return deltaInput;
    }

    protected void iterateForward(Tensor input, float[] outData,
                                  int numChannels, int outHeight, int outWidth) {
        for (int sampleIndex = 0; sampleIndex < batchSize; sampleIndex++) {
            for (int chanelIndex = 0; chanelIndex < numChannels; chanelIndex++) {
                for (int rowId = 0; rowId < outHeight; rowId++) {
                    int startY = rowId * this.height;
                    for (int colId = 0; colId < outWidth; colId++) {
                        int startX = colId * this.width;
                        int offset = this.computeOffset(sampleIndex, chanelIndex, rowId, colId, outHeight, outWidth, numChannels);
                        outData[offset] = computePooling(
                                input, sampleIndex, chanelIndex, startY, startX, this.height, this.width, this.padHeight, this.padWidth
                        );
                    }
                }
            }
        }
    }

    protected void iterateBackward(float[] deltaData, float[] deltaInput,
                                   int numChannels, int outH, int outW) {
        for (int sampleIndex = 0; sampleIndex < this.batchSize; sampleIndex++) {
            for (int channelIndex = 0; channelIndex < numChannels; channelIndex++) {
                for (int y = 0; y < outH; y++) {
                    int startY = y * this.height;
                    for (int x = 0; x < outW; x++) {
                        int startX = x * this.width;
                        int offset = computeOffset(sampleIndex, channelIndex, y, x, outH, outW, numChannels);
                        distributeGradient(deltaInput, input.getData(), sampleIndex, channelIndex,
                                startY, startX, this.height, this.width, this.padHeight, this.padWidth, deltaData[offset]);
                    }
                }
            }
        }
    }

    protected float computePooling(Tensor input, int sampleId, int channelId,
                                    int startY, int startX,
                                    int poolHeight, int poolWidth,
                                    int padHeight, int padWidth) {
        float result = getIdentityValue();
        float[] data = input.getData();
        int[] inputShape = input.getShape();
        int[] inputStrides = input.getStrides();

        for (int row = 0; row < poolHeight; row++) {
            for (int col = 0; col < poolWidth; col++) {
                int y = startY + row - padHeight;
                int x = startX + col - padWidth;
                if (y < 0 || y >= inputShape[1] || x < 0 || x >= inputShape[2]) continue;

                int offset = sampleId * inputStrides[0]
                        + y * inputStrides[1]
                        + x * inputStrides[2]
                        + channelId * inputStrides[3];
                result = poolingFunction(result, data[offset]);
            }
        }
        return result;
    }

    protected int computeOffset(int sampleIndex, int channelIndex, int y, int x, int outH, int outW, int channels) {
        return sampleIndex * (outH * outW * channels) + y * (outW * channels) + x * channels + channelIndex;
    }
}
