package io.h3ca.forgemind.core.layers;

import com.fasterxml.jackson.annotation.JsonTypeName;
import io.h3ca.forgemind.core.api.Tensor;
import tools.jackson.databind.annotation.JsonDeserialize;
import tools.jackson.databind.annotation.JsonPOJOBuilder;

import java.util.Arrays;

@JsonTypeName("Flatten")
@JsonDeserialize(builder = Flatten.Builder.class)
public class Flatten extends Layer {

    private Tensor input;

    private Flatten() {
        super();
    }

    @JsonTypeName("FlattenBuilder")
    @JsonPOJOBuilder(withPrefix = "")
    public static class Builder implements LayerBuilder {

        public Builder() {
            super();
        }

        @Override
        public Layer build() {
            return new Flatten();
        }
    }

    @Override
    public Tensor forward(Tensor input) {
        this.input = input;

        int[] inputShape = this.input.getShape();
        int flatSize = Arrays.stream(inputShape).skip(1).reduce(1, Math::multiplyExact);
        this.cache.ensureOutput(new int[]{inputShape[0], flatSize}, this.input.getData());

        return this.cache.getOutput();
    }

    @Override
    public Tensor backward(Tensor deltaOutput) {
        int[] inputShape = this.input.getShape();
        this.cache.ensureInputDelta(inputShape, deltaOutput.getData());
        return this.cache.getInputDelta();
    }
}
