package io.h3ca.core.layers;

import io.h3ca.forgemind.core.api.Initializer;
import io.h3ca.forgemind.core.api.Tensor;
import io.h3ca.forgemind.core.layers.Dense;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;

class DenseTest {

    private final static float DELTA = 1e-6f;

    @Test
    void forwardSimple() {
        int inputSize = 2;
        int outputSize = 2;
        int initialWeight = 2;
        Dense dense = (Dense) new Dense.Builder(outputSize)
                .initializer(new Initializer.Constant(initialWeight))
                .build();

        Tensor input = new Tensor(new float[]{1, 3}, new int[]{1, inputSize});
        assertEquals(inputSize, input.size());

        Tensor output = dense.forward(input);
        assertEquals(outputSize, output.size());

        float[] expected = new float[]{8, 8};
        assertArrayEquals(
                expected,
                output.getData(),
                DELTA
        );
    }

    @Test
    void forwardExpandFeatures() {
        int inputSize = 2;
        int outputSize = 10;
        int initialWeight = 2;
        int biasValue = 0;
        Dense dense = (Dense) new Dense.Builder(outputSize)
                .initializer(new Initializer.Constant(initialWeight))
                .build();

        float[] inputData = new float[]{1, 3};
        Tensor input = new Tensor(inputData, new int[]{1, inputSize});
        assertEquals(inputSize, input.size());

        Tensor output = dense.forward(input);
        assertEquals(outputSize, output.size());

        float[] expected = new float[outputSize];
        for (int i = 0; i < outputSize; i++) {
            for (float inputDatum : inputData) {
                expected[i] += inputDatum * initialWeight + biasValue;
            }
        }

        assertArrayEquals(
                expected,
                output.getData(),
                DELTA
        );
    }

    @Test
    void forwardCompressFeatures() {
        int inputSize = 5;
        int outputSize = 2;
        int initialWeight = 2;
        int biasValue = 0;
        Dense dense = (Dense) new Dense.Builder(outputSize)
                .initializer(new Initializer.Constant(initialWeight))
                .build();

        float[] inputData = new float[]{1, 2, 3, 4, 5};
        Tensor input = new Tensor(inputData, new int[]{1, inputSize});
        assertEquals(inputSize, input.size());

        Tensor output = dense.forward(input);
        assertEquals(outputSize, output.size());

        float[] expected = new float[outputSize];
        for (int i = 0; i < outputSize; i++) {
            for (float inputDatum : inputData) {
                expected[i] += inputDatum * initialWeight + biasValue;
            }
        }

        assertArrayEquals(
                expected,
                output.getData(),
                DELTA
        );
    }

    @Test
    void backwardSimple() {
        int inputSize = 2;
        int outputSize = 2;
        int initialWeight = 2;
        Dense dense = (Dense) new Dense.Builder(outputSize)
                .initializer(new Initializer.Constant(initialWeight))
                .build();

        Tensor input = new Tensor(new float[]{1, 3}, new int[]{1, inputSize});
        assertEquals(inputSize, input.size());

        Tensor output = dense.forward(input);
        assertEquals(outputSize, output.size());

        Tensor gradOutput = new Tensor(new float[]{1, 1}, new int[]{1, 2});
        assertEquals(inputSize, gradOutput.size());

        Tensor gradInput = dense.backward(gradOutput);
        assertEquals(inputSize, gradInput.size());

        float[] expectedGradInput = new float[]{4, 4};

        assertArrayEquals(expectedGradInput, gradInput.getData(), DELTA);
    }

    @Test
    void backwardExpandFeatures() {
        int inputSize = 2;
        int outputSize = 5;
        int initialWeight = 2;
        Dense dense = (Dense) new Dense.Builder(outputSize)
                .initializer(new Initializer.Constant(initialWeight))
                .build();

        Tensor input = new Tensor(new float[]{1, 3}, new int[]{1, inputSize});
        assertEquals(inputSize, input.size());

        Tensor output = dense.forward(input);
        assertEquals(outputSize, output.size());

        Tensor gradOutput = new Tensor(new float[]{1, 1, 1, 1, 1}, new int[]{1, outputSize});
        assertEquals(outputSize, gradOutput.size());

        Tensor gradInput = dense.backward(gradOutput);
        assertEquals(inputSize, gradInput.size());

        float[] expectedGradInput = new float[]{10, 10};

        assertArrayEquals(expectedGradInput, gradInput.getData(), DELTA);
    }

    @Test
    void backwardCompressFeatures() {
        int inputSize = 5;
        int outputSize = 2;
        int initialWeight = 2;
        Dense dense = (Dense) new Dense.Builder(outputSize)
                .initializer(new Initializer.Constant(initialWeight))
                .build();

        Tensor input = new Tensor(new float[]{1, 2, 3, 4, 5}, new int[]{1, inputSize});
        dense.forward(input);

        Tensor gradOutput = new Tensor(new float[]{1, 1}, new int[]{1, outputSize});
        assertEquals(outputSize, gradOutput.size());

        Tensor gradInput = dense.backward(gradOutput);
        assertEquals(inputSize, gradInput.size());

        float[] expectedGradInput = new float[]{4, 4, 4, 4, 4};

        assertArrayEquals(expectedGradInput, gradInput.getData(), DELTA);
    }
}