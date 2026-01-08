package io.h3ca.forgemind.core.loss;

import com.fasterxml.jackson.annotation.JsonSubTypes;
import com.fasterxml.jackson.annotation.JsonTypeInfo;
import io.h3ca.forgemind.core.api.Tensor;

@JsonTypeInfo(
        use = JsonTypeInfo.Id.NAME,
        include = JsonTypeInfo.As.PROPERTY,
        property = "type"
)
@JsonSubTypes({
        @JsonSubTypes.Type(value = HuberLoss.class, name = "HuberLoss"),
        @JsonSubTypes.Type(value = CrossEntropy.class, name = "CrossEntropy"),
        @JsonSubTypes.Type(value = L1Loss.class, name = "L1Loss")
})
public interface LossFunction {

    float computeLoss(Tensor yBatch, Tensor predictionBatch);
    Tensor computeDelta(Tensor yBatch, Tensor predictionBatch);
}
