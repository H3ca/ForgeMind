package io.h3ca.core.layers;

import io.h3ca.forgemind.core.api.Initializer;
import io.h3ca.forgemind.core.api.Tensor;
import io.h3ca.forgemind.core.layers.Conv;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

public class ConvTest {

    private final static float DELTA = 1e-6f;

    private final int val = 5;
    private final float[] inputData = new float[]{
            this.val, this.val, this.val, this.val, this.val,
            this.val, this.val, this.val, this.val, this.val,
            this.val, this.val, this.val, this.val, this.val,
            this.val, this.val, this.val, this.val, this.val,
            this.val, this.val, this.val, this.val, this.val
    };
    private final int inputSize = (int) Math.sqrt(this.inputData.length);

    private final Tensor input = new Tensor(this.inputData, new int[]{1, this.inputSize, this.inputSize, 1});

    private final int numFilters = 1;
    private final int kernelSize = 3;
    private final int[] kernelShape = new int[]{this.kernelSize, this.kernelSize};

    private final float kernelValue = (float ) 1 / (this.kernelSize * this.kernelSize);
    private final float valExp1 = (this.val * 4) * this.kernelValue;
    private final float valExp2 = (this.val * 6) * this.kernelValue;
    private final float valExp3 = (this.val * 9) * this.kernelValue;

    private final float[] expectedOutputData = new float[]{
            this.valExp1, this.valExp2, this.valExp2, this.valExp2, this.valExp1,
            this.valExp2, this.valExp3, this.valExp3, this.valExp3, this.valExp2,
            this.valExp2, this.valExp3, this.valExp3, this.valExp3, this.valExp2,
            this.valExp2, this.valExp3, this.valExp3, this.valExp3, this.valExp2,
            this.valExp1, this.valExp2, this.valExp2, this.valExp2, this.valExp1,
    };

    @Test
    void forwardSimple() {
        Conv conv = (Conv) new Conv.Builder(this.numFilters, this.kernelShape)
                .initializer(new Initializer.Constant(this.kernelValue))
                .build();

        Tensor output = conv.forward(this.input);
        int[] outputShape = output.getShape();
        assertEquals(this.numFilters, outputShape[3]);

        assertArrayEquals(
                this.expectedOutputData,
                output.getData(),
                DELTA
        );
    }

    @Test
    void forwardExpandFeatures() {
        int numFilters = 5;
        Conv conv = (Conv) new Conv.Builder(numFilters, this.kernelShape)
                .initializer(new Initializer.Constant(this.kernelValue))
                .build();

        Tensor output = conv.forward(this.input);
        int[] inputShape = this.input.getShape();
        int[] outputShape = output.getShape();
        assertEquals(inputShape[0], outputShape[0]);
        assertEquals(inputShape[1], outputShape[1]);
        assertEquals(inputShape[2], outputShape[2]);
        assertEquals(numFilters, outputShape[3]);

        int newLength = this.expectedOutputData.length * numFilters;

        float[] expandedExpectedOutputData = new float[newLength];
        for (int i = 0; i < this.expectedOutputData.length; i++) {
            for (int j = 0; j < numFilters; j++) {
                expandedExpectedOutputData[i * numFilters + j] = this.expectedOutputData[i];
            }
        }

        assertArrayEquals(
                expandedExpectedOutputData,
                output.getData(),
                DELTA
        );
    }

    @Test
    void backwardSimple() {
        Conv conv = (Conv) new Conv.Builder(this.numFilters, this.kernelShape)
                .initializer(new Initializer.Constant(this.kernelValue))
                .build();

        Tensor output = conv.forward(this.input);
        int[] outputShape = output.getShape();
        assertEquals(this.numFilters, outputShape[3]);

        assertArrayEquals(
                this.expectedOutputData,
                output.getData(),
                DELTA
        );

        int gradExtra = 1;
        float[] gradDataOutput = new float[this.inputData.length];
        for (int i = 0; i < this.inputData.length; i++) {
            gradDataOutput[i] = this.inputData[i] + gradExtra;
        }

        float valExp1 = ((this.val + gradExtra)  * 4) * this.kernelValue;
        float valExp2 = ((this.val + gradExtra) * 6) * this.kernelValue;
        float valExp3 = ((this.val + gradExtra) * 9) * this.kernelValue;

        float[] expectedGradData = new float[]{
                valExp1, valExp2, valExp2, valExp2, valExp1,
                valExp2, valExp3, valExp3, valExp3, valExp2,
                valExp2, valExp3, valExp3, valExp3, valExp2,
                valExp2, valExp3, valExp3, valExp3, valExp2,
                valExp1, valExp2, valExp2, valExp2, valExp1,
        };

        Tensor gradOutput = new Tensor(gradDataOutput, this.input.getShape());
        Tensor gradInput = conv.backward(gradOutput);
        assertArrayEquals(
                expectedGradData,
                gradInput.getData(),
                DELTA
        );
    }
}
