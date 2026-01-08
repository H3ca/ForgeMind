package io.h3ca.forgemind.core.optimizers;

import com.fasterxml.jackson.annotation.JsonSubTypes;
import com.fasterxml.jackson.annotation.JsonTypeInfo;

@JsonTypeInfo(
        use = JsonTypeInfo.Id.NAME,
        include = JsonTypeInfo.As.PROPERTY,
        property = "type"
)
@JsonSubTypes({
        @JsonSubTypes.Type(value = SGD.class, name = "SGD"),
        @JsonSubTypes.Type(value = Adam.class, name = "Adam")
})
public interface Optimizer {

    StateInterface createState();
    void apply(StateInterface stateInterface, float[] parameters, float[] gradients);

    interface StateInterface {
        void initialize(int size);
    }

    class EmptyState implements StateInterface {
        private static final EmptyState INSTANCE = new EmptyState();

        private EmptyState() {}

        public static EmptyState get() {
            return INSTANCE;
        }

        @Override
        public void initialize(int size) {}
    }
}
