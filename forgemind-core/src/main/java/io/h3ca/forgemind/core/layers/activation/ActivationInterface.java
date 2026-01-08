package io.h3ca.forgemind.core.layers.activation;

public interface ActivationInterface {

    float activate(float x);

    /**
     * @param z the output of the activation function (i.e., {@code activate(x)})
     * @return the derivative of the activation function evaluated at that point
     */
    float derivative(float z);
}
