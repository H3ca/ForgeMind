package io.h3ca.forgemind.core.callbacks;

import com.fasterxml.jackson.annotation.JsonProperty;
import tools.jackson.databind.annotation.JsonDeserialize;
import tools.jackson.databind.annotation.JsonPOJOBuilder;

@JsonDeserialize(builder = EarlyStopping.Builder.class)
public class EarlyStopping {

    @JsonProperty("patience")
    private final int patience;
    @JsonProperty("minDelta")
    private final float minDelta;
    @JsonProperty("restoreBestWeights")
    private final boolean restoreBestWeights;

    private int patienceCounter = 0;
    private float bestLoss = Float.POSITIVE_INFINITY;

    private EarlyStopping(int patience, float minDelta, boolean restoreBestWeights) {
        if (patience < 0) throw new IllegalArgumentException("patience can't be negative.");
        if (minDelta < 0) throw new IllegalArgumentException("minDelta can't be negative.");
        this.patience = patience;
        this.minDelta = minDelta;
        this.restoreBestWeights = restoreBestWeights;
    }

    public boolean update(float loss) {
        if (this.bestLoss - loss > this.minDelta) {
            this.bestLoss = loss;
            this.patienceCounter = 0;
            return true;
        }

        this.patienceCounter++;
        return false;
    }

    public boolean shouldStop() {
        return this.patienceCounter >= this.patience;
    }

    public boolean shouldRestoreBestWeights() {
        return this.restoreBestWeights;
    }

    @JsonPOJOBuilder(withPrefix = "")
    public static class Builder {
        private int patience = 0;
        private float minDelta = 0;
        private boolean restoreBestWeights = false;

        public Builder patience(int patience) {
            if (patience < 0) throw new IllegalArgumentException("patience cannot be negative.");
            this.patience = patience;
            return this;
        }

        public Builder minDelta(float minDelta) {
            if (minDelta < 0) throw new IllegalArgumentException("minDelta cannot be negative.");
            this.minDelta = minDelta;
            return this;
        }

        public Builder restoreBestWeights(boolean restoreBestWeights) {
            this.restoreBestWeights = restoreBestWeights;
            return this;
        }

        public EarlyStopping build() {
            return new EarlyStopping(this.patience, this.minDelta, this.restoreBestWeights);
        }
    }
}
