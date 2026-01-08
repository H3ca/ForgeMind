package io.h3ca.forgemind.core.layers.pooling;

public class SumPooling extends Pooling {

    public SumPooling(int width, int height, int padWidth, int padHeight) {
        super(width, height, padWidth, padHeight);
    }

    @Override
    protected float poolingFunction(float a, float b) {
        return a + b;
    }

    @Override
    protected float getIdentityValue() {
        return 0.0f;
    }

    @Override
    protected void distributeGradient(float[] deltaInput, float[] inputData,
                                      int sampleId, int channelId,
                                      int startY, int startX,
                                      int poolHeight, int poolWidth,
                                      int padHeight, int padWidth,
                                      float grad) {

        int[] inputShape = this.input.getShape();
        int[] inputStrides = this.input.getStrides();

        for (int row = 0; row < poolHeight; row++) {
            int y = startY + row - padHeight;
            for (int col = 0; col < poolWidth; col++) {
                int x = startX + col - padWidth;
                if (y >= 0 && y < inputShape[1] && x >= 0 && x < inputShape[2]) {
                    int offset = sampleId * inputStrides[0]
                            + y * inputStrides[1]
                            + x * inputStrides[2]
                            + channelId * inputStrides[3];
                    deltaInput[offset] += grad;
                }
            }
        }
    }
}
