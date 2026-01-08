package io.h3ca.forgemind.core.layers.activation;

public enum ActivationFunction implements ActivationInterface {
    Sigmoid {
        @Override
        public float activate(float x) {
            return (float) (1 / (1 + Math.exp(-x)));
        }

        @Override
        public float derivative(float z) {
            return z * (1 - z);
        }
    },
    ReLu {
        @Override
        public float activate(float x) {
            return Math.max(0, x);
        }

        @Override
        public float derivative(float z) {
            return z > 0 ? 1 : 0;
        }
    }
}
