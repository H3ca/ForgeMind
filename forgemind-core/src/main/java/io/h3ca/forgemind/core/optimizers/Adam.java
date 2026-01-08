package io.h3ca.forgemind.core.optimizers;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.JsonTypeName;
import tools.jackson.databind.annotation.JsonDeserialize;
import tools.jackson.databind.annotation.JsonPOJOBuilder;

@JsonTypeName("Adam")
@JsonDeserialize(builder = Adam.Builder.class)
public class Adam implements Optimizer {

    @JsonProperty("learningRate")
    private final float learningRate;
    @JsonProperty("beta1")
    private final float beta1;
    @JsonProperty("beta2")
    private final float beta2;
    @JsonProperty("epsilon")
    private final float epsilon;

    private float beta1t;
    private float beta2t;

    private Adam(float learningRate, float beta1, float beta2, float epsilon) {
        if (learningRate <= 0) throw new IllegalArgumentException("learningRate must be positive.");
        if (beta1 < 0 || beta1 >= 1) throw new IllegalArgumentException("beta1 must be between 0 and 1.");
        if (beta2 < 0 || beta2 >= 1) throw new IllegalArgumentException("beta2 must be between 0 and 1.");
        if (epsilon < 0) throw new IllegalArgumentException("epsilon must positive.");

        this.learningRate = learningRate;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.beta1t = beta1;
        this.beta2t = beta2;
        this.epsilon = epsilon;
    }

    @Override
    public StateInterface createState() {
        return new Adam.State();
    }

    @Override
    public void apply(StateInterface stateInterface, float[] parameters, float[] gradients) {
        Adam.State state = (Adam.State) stateInterface;
        this.beta1t *= this.beta1;
        this.beta2t *= this.beta2;

        for (int i = 0; i < parameters.length; i++) {
            state.mean[i] = this.beta1 * state.mean[i] + (1 - this.beta1) * gradients[i];
            state.variance[i] = this.beta2 * state.variance[i] + (1 - this.beta2) * (gradients[i] * gradients[i]);

            float meanCorrection = state.mean[i] / (1 - this.beta1t);
            float varianceCorrection = state.variance[i] / (1 - this.beta2t);

            parameters[i] -= this.learningRate * (float)(meanCorrection / (Math.sqrt(varianceCorrection) + this.epsilon));
        }
    }

    @JsonPOJOBuilder(withPrefix = "")
    public static class Builder {
        private float learningRate = 1e-3f;
        private float beta1 = 0.9f;
        private float beta2 = 0.999f;
        private float epsilon = 1e-9f;

        public Builder learningRate(float learningRate) {
            if (learningRate <= 0) throw new IllegalArgumentException("learningRate must be positive.");
            this.learningRate = learningRate;
            return this;
        }

        public Builder beta1(float beta1) {
            if (beta1 < 0 || beta1 >= 1) throw new IllegalArgumentException("beta1 must be between 0 and 1.");
            this.beta1 = beta1;
            return this;
        }

        public Builder beta2(float beta2) {
            if (beta2 < 0 || beta2 >= 1) throw new IllegalArgumentException("beta2 must be between 0 and 1.");
            this.beta2 = beta2;
            return this;
        }

        public Builder epsilon(float epsilon) {
            if (epsilon < 0) throw new IllegalArgumentException("epsilon must positive.");
            this.epsilon = epsilon;
            return this;
        }

        public Optimizer build() {
            return new Adam(this.learningRate, this.beta1, this.beta2, this.epsilon);
        }
    }

    public static class State implements StateInterface {

        public float[] mean;
        public float[] variance;

        public State() {}

        public void initialize(int size) {
            if (size <= 0) throw new IllegalArgumentException("size must be > 0.");

            this.mean = new float[size];
            this.variance = new float[size];
        }
    }
}
