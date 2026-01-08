package io.h3ca.forgemind.core.utils;

import io.h3ca.forgemind.core.api.Tensor;
import io.h3ca.forgemind.core.loss.LossFunction;

public interface Metric {

    float compute(Tensor yTrue, Tensor yPred);

    public static class MSE implements Metric {
        private final LossFunction lossFunction;

        public MSE(LossFunction lossFunction) {
            this.lossFunction = lossFunction;
        }


        @Override
        public float compute(Tensor yTrue, Tensor yPred) {
            Tensor delta = this.lossFunction.computeDelta(yTrue, yPred);
            return Metrics.getMSE(delta);
        }
    }
}
