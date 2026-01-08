package io.h3ca.forgemind.core.api;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import io.h3ca.forgemind.core.layers.LayerBuilder;

import java.util.Arrays;

public class ModelConfig {

        private final LayerBuilder[] layers;
        private final int size;

        private ModelConfig(LayerBuilder[] layers) {
            this.layers = Arrays.copyOf(layers, layers.length);
            this.size = this.layers.length;
        }

        @JsonCreator
        public static ModelConfig fromJson(@JsonProperty("layers") LayerBuilder[] layers) {
            return new ModelConfig(layers);
        }

        public static ModelConfig of(LayerBuilder... layers) {
            return new ModelConfig(layers);
        }

        public LayerBuilder[] getLayers() {
            return this.layers;
        }

        public int getNumberOfLayers() {
        return this.size;
    }
}
