module io.h3ca.forgemind {
    exports io.h3ca.forgemind.core.api;
    exports io.h3ca.forgemind.core.callbacks;
    exports io.h3ca.forgemind.core.layers;
    exports io.h3ca.forgemind.core.layers.activation;
    exports io.h3ca.forgemind.core.layers.pooling;
    exports io.h3ca.forgemind.core.loss;
    exports io.h3ca.forgemind.core.optimizers;
    exports io.h3ca.forgemind.core.utils;

    exports io.h3ca.forgemind.datasets;

    requires com.fasterxml.jackson.annotation;
    requires java.datatransfer;
    requires java.desktop;
    requires org.slf4j;
    requires tools.jackson.databind;

    opens io.h3ca.forgemind.core.api to tools.jackson.databind;
    opens io.h3ca.forgemind.core.callbacks to tools.jackson.databind;
    opens io.h3ca.forgemind.core.layers to tools.jackson.databind;
    opens io.h3ca.forgemind.core.layers.activation to tools.jackson.databind;
    opens io.h3ca.forgemind.core.layers.pooling to tools.jackson.databind;
    opens io.h3ca.forgemind.core.loss to tools.jackson.databind;
    opens io.h3ca.forgemind.core.optimizers to tools.jackson.databind;
}