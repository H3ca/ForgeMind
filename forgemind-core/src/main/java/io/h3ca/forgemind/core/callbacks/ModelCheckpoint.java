package io.h3ca.forgemind.core.callbacks;

import com.fasterxml.jackson.annotation.JsonProperty;
import io.h3ca.forgemind.core.api.Model;
import tools.jackson.databind.annotation.JsonPOJOBuilder;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Objects;

public class ModelCheckpoint {

    private final Model model;
    private final Path checkpointPath;
    private final boolean isTemp;
    private boolean hasCheckpoint = false;

    private ModelCheckpoint(Model model, Path checkpointPath) {
        this.model = Objects.requireNonNull(model, "model cannot be null");
        this.checkpointPath = (checkpointPath != null) ? checkpointPath : createTempCheckpoint();
        this.isTemp = (checkpointPath == null);
    }

    @JsonPOJOBuilder(withPrefix = "")
    public static class Builder {
        private Model model;
        @JsonProperty("checkpointPath")
        private Path checkpointPath;
        private boolean used = false;

        public Builder model(Model model) {
            if (model == null) throw new IllegalArgumentException("model cannot be null.");
            this.model = model;
            return this;
        }

        public Builder checkpointPath(Path checkpointPath) {
            this.checkpointPath = checkpointPath;
            return this;
        }
        public ModelCheckpoint build() {
            if (this.used) throw new IllegalStateException("ModelCheckpoint.Builder can only be used once.");
            if (this.model == null) throw new IllegalStateException("The model was not provided.");

            this.used = true;
            return new ModelCheckpoint(this.model, this.checkpointPath);
        }
    }

    public void onNewBest() {
        try {
            this.model.save(this.checkpointPath);
        } catch (RuntimeException e) {
            throw new RuntimeException("Failed to save model checkpoint", e);
        }
        this.hasCheckpoint = true;
    }

    public void onTrainingEnd() {
        if (!this.hasCheckpoint) return;

        Model best = Model.load(this.checkpointPath);
        this.model.copyWeightsFrom(best);
        this.deleteCheckpoint();
    }

    private Path createTempCheckpoint() {
        try {
            Path temp = Files.createTempFile("best_model", ".fmind");
            temp.toFile().deleteOnExit();
            return temp;
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    private void deleteCheckpoint() {
        if (!this.isTemp) return;
        try {
            Files.deleteIfExists(this.checkpointPath);
        } catch (IOException ignored) {}
    }
}
