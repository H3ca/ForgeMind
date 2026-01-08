package io.h3ca.forgemind.core.utils;

import io.h3ca.forgemind.core.api.Tensor;

public class Metrics {

    public static float getMSE(Tensor batchErrors) {
        float mse = 0;
        float[] errorsData = batchErrors.getData();
        for (float error : errorsData) {
            mse += error * error;
        }
        mse /= batchErrors.getShape()[0] * batchErrors.getShape()[1];
        return mse;
    }
}
