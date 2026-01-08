package io.h3ca.forgemind.core.optimizers;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.JsonTypeName;

@JsonTypeName("SGD")
public class SGD implements Optimizer {

    @JsonProperty("learningRate")
    private final float learningRate;

    public SGD(float learningRate) {
        if (learningRate <= 0) throw new IllegalArgumentException("learningRate must be positive");
        this.learningRate = learningRate;
    }

    public SGD() {
        this(0.01f);
    }

    @Override
    public EmptyState createState() {
        return EmptyState.get();
    }

    @Override
    public void apply(StateInterface stateInterface, float[] parameters, float[] gradients) {
        for (int biasId = 0; biasId < parameters.length; biasId++) {
            parameters[biasId] -= this.learningRate * gradients[biasId];
        }
    }
}
