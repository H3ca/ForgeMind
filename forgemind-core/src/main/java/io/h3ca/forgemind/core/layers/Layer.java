package io.h3ca.forgemind.core.layers;

import com.fasterxml.jackson.annotation.JsonTypeInfo;
import io.h3ca.forgemind.core.api.Tensor;
import io.h3ca.forgemind.core.optimizers.Optimizer;

@JsonTypeInfo(
        use = JsonTypeInfo.Id.NAME,
        include = JsonTypeInfo.As.PROPERTY,
        property = "type"
)
public abstract class Layer {

    // Number of output features per sample
    protected int batchSize;
    protected LayerCache cache;

    public Layer() {
        this.cache = new LayerCache();
    }

    public abstract Tensor forward(Tensor input);

    public abstract Tensor backward(Tensor deltaOutput);

    public void update(Optimizer optimizer) {}

    public void copyParametersFrom(Layer other) {}
}
