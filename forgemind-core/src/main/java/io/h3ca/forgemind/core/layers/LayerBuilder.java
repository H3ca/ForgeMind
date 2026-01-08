package io.h3ca.forgemind.core.layers;

import com.fasterxml.jackson.annotation.JsonSubTypes;
import com.fasterxml.jackson.annotation.JsonTypeInfo;
import io.h3ca.forgemind.core.layers.activation.Activation;
import io.h3ca.forgemind.core.layers.activation.Softmax;
import io.h3ca.forgemind.core.layers.pooling.Pooling;

@JsonTypeInfo(
        use = JsonTypeInfo.Id.NAME,
        include = JsonTypeInfo.As.PROPERTY,
        property = "type"
)
@JsonSubTypes({
        @JsonSubTypes.Type(value = Conv.Builder.class, name = "ConvBuilder"),
        @JsonSubTypes.Type(value = Dense.Builder.class, name = "DenseBuilder"),
        @JsonSubTypes.Type(value = Flatten.Builder.class, name = "FlattenBuilder"),
        @JsonSubTypes.Type(value = Pooling.Builder.class, name = "PoolingBuilder"),
        @JsonSubTypes.Type(value = Softmax.Builder.class, name = "SoftmaxBuilder"),
        @JsonSubTypes.Type(value = Activation.Builder.class, name = "ActivationBuilder"),
})
public interface LayerBuilder {

    public Layer build();
}
