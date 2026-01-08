package io.h3ca.forgemind.core.layers.pooling;

import io.h3ca.forgemind.core.api.Tensor;

public class AvgPooling extends Pooling {

    // Stores number of valid elements per output position
    private int[] validCounts;

    public AvgPooling(int width, int height, int padWidth, int padHeight) {
        super(width, height, padWidth, padHeight);
    }

    // Required by base class, but unused
    @Override
    protected float poolingFunction(float a, float b) {
        return 0.0f;
    }

    @Override
    protected float getIdentityValue() {
        return 0.0f;
    }

    @Override
    protected float computePooling(Tensor input, int sampleId, int channelId,
                                    int startY, int startX,
                                    int poolHeight, int poolWidth,
                                    int padHeight, int padWidth) {

        float sum = 0.0f;
        int count = 0;
        float[] inputData = input.getData();
        int[] inputShape = input.getShape();
        int[] inputStrides = input.getStrides();

        for (int row = 0; row < poolHeight; row++) {
            int y = startY + row - padHeight;
            for (int col = 0; col < poolWidth; col++) {
                int x = startX + col - padWidth;
                if (y < 0 || y > inputShape[1] || x < 0 || x > inputShape[2]) continue;

                int inputOffset = sampleId * inputStrides[0]
                        + y * inputStrides[1]
                        + x * inputStrides[2]
                        + channelId * inputStrides[3];
                sum += inputData[inputOffset];
                count++;
            }
        }

        int y = startY / this.height;
        int x = startX / this.width;
        int[] featuresStrides = this.cache.getOutput().getStrides();

        // Store count for backward
        int outOffset = sampleId * featuresStrides[0]
                + y * featuresStrides[1]
                + x * featuresStrides[2]
                + channelId * featuresStrides[3];

        if (this.validCounts == null || this.validCounts.length != this.cache.getOutput().getData().length) {
            this.validCounts = new int[this.cache.getOutput().getData().length];
        }
        this.validCounts[outOffset] = count;

        return count > 0 ? sum / count : 0.0f;
    }

    @Override
    protected void distributeGradient(float[] deltaInput, float[] inputData,
                                      int sampleId, int channelId,
                                      int startY, int startX,
                                      int poolHeight, int poolWidth,
                                      int padHeight, int padWidth,
                                      float grad) {

        int[] featuresStrides = this.cache.getOutput().getStrides();

        int yOut = startY / this.height;
        int xOut = startX / this.width;

        int outOffset = sampleId * featuresStrides[0]
                + yOut * featuresStrides[1] +
                xOut * featuresStrides[2] +
                channelId * featuresStrides[3];

        int count = this.validCounts[outOffset];
        if (count == 0) return;

        float gradPer = grad / count;

        int[] inputShape = this.input.getShape();
        int[] inputStrides = this.input.getStrides();

        for (int row = 0; row < poolHeight; row++) {
            int y = startY + row - padHeight;
            for (int col = 0; col < poolWidth; col++) {
                int x = startX + col - padWidth;
                if (y >= 0 && y < inputShape[1] && x >= 0 && x < inputShape[2]) {
                    int deltaOffset = sampleId * inputStrides[0]
                            + y * inputStrides[1]
                            + x * inputStrides[2]
                            + channelId * inputStrides[3];
                    deltaInput[deltaOffset] += gradPer;
                }
            }
        }
    }
}
