package io.h3ca.forgemind.core.api;

import com.fasterxml.jackson.annotation.JsonProperty;
import io.h3ca.forgemind.core.callbacks.ModelCheckpoint;
import io.h3ca.forgemind.core.callbacks.EarlyStopping;
import io.h3ca.forgemind.core.layers.LayerBuilder;
import io.h3ca.forgemind.core.layers.Layer;
import io.h3ca.forgemind.core.loss.CrossEntropy;
import io.h3ca.forgemind.core.loss.LossFunction;
import io.h3ca.forgemind.core.optimizers.Adam;
import io.h3ca.forgemind.core.optimizers.Optimizer;
import io.h3ca.forgemind.core.utils.ModelSerializer;
import io.h3ca.forgemind.core.utils.Metric;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import tools.jackson.databind.annotation.JsonDeserialize;
import tools.jackson.databind.annotation.JsonPOJOBuilder;

import java.nio.file.Path;
import java.util.Objects;

@JsonDeserialize(builder = Model.Builder.class)
public class Model {
    private static final Logger LOG = LoggerFactory.getLogger(Model.class);

    @JsonProperty("layers")
    private final Layer[] layers;
    @JsonProperty("config")
    private final ModelConfig modelConfig;
    @JsonProperty("lossFunction")
    private final LossFunction lossFunction;
    @JsonProperty("optimizer")
    private final Optimizer optimizer;
    @JsonProperty("earlyStopping")
    private final EarlyStopping earlyStopping;
    @JsonProperty("checkpointBuilder")
    private final ModelCheckpoint.Builder checkpointBuilder;
    private final ModelCheckpoint modelCheckpoint;
    private final Metric mse;

    private Model(LossFunction lossFunction, Optimizer optimizer, EarlyStopping earlyStopping,
                  ModelCheckpoint.Builder checkpointBuilder, ModelConfig config) {

        this.lossFunction = Objects.requireNonNull(lossFunction, "lossFunction cannot be null");
        this.optimizer = Objects.requireNonNull(optimizer, "optimizer cannot be null");
        this.earlyStopping = Objects.requireNonNull(earlyStopping, "earlyStopping cannot be null");
        this.checkpointBuilder = Objects.requireNonNull(checkpointBuilder, "checkpointBuilder cannot be null");
        this.modelConfig = Objects.requireNonNull(config, "config cannot be null");

        this.modelCheckpoint = checkpointBuilder.model(this).build();
        this.mse = new Metric.MSE(this.lossFunction);
        this.layers = new Layer[config.getNumberOfLayers()];

        LayerBuilder[] layerConfig = this.modelConfig.getLayers();
        for (int layerId = 0; layerId < config.getNumberOfLayers(); layerId++) {
            if (layerConfig[layerId] == null)
                throw new IllegalArgumentException("LayerBuilder at index " + layerId + " cannot be null.");
            this.layers[layerId] = layerConfig[layerId].build();
        }
    }

    @JsonPOJOBuilder(withPrefix = "")
    public static class Builder {
        private LossFunction lossFunction = CrossEntropy.get();
        private Optimizer optimizer = new Adam.Builder().build();
        private EarlyStopping earlyStopping = new EarlyStopping.Builder().build();
        private ModelCheckpoint.Builder checkpointBuilder = new ModelCheckpoint.Builder();
        private ModelConfig config;

        public Builder lossFunction(LossFunction lossFunction) {
            if (lossFunction == null) throw new IllegalArgumentException("lossFunction cannot be null.");
            this.lossFunction = lossFunction;
            return this;
        }

        public Builder optimizer(Optimizer optimizer) {
            if (optimizer == null) throw new IllegalArgumentException("optimizer cannot be null.");
            this.optimizer = optimizer;
            return this;
        }

        public Builder earlyStopping(EarlyStopping earlyStopping) {
            if (earlyStopping == null) throw new IllegalArgumentException("earlyStopping cannot be null.");
            this.earlyStopping = earlyStopping;
            return this;
        }

        @JsonProperty("config")
        public Builder config(ModelConfig config) {
            if (config == null) throw new IllegalArgumentException("config cannot be null.");
            this.config = config;
            return this;
        }

        public Builder config(LayerBuilder... layersConfig) {
            if (layersConfig == null) throw new IllegalArgumentException("layersConfig cannot be null.");
            this.config = ModelConfig.of(layersConfig);
            return this;
        }

        public Builder checkpointBuilder(ModelCheckpoint.Builder checkpointBuilder) {
            if (checkpointBuilder == null) throw new IllegalArgumentException("checkpointBuilder cannot be null.");
            this.checkpointBuilder = checkpointBuilder;
            return this;
        }

        public Model build() {
            if (this.config == null)
                throw new IllegalStateException("You forgot to provide required configuration before building the model.");
            return new Model(this.lossFunction, this.optimizer, this.earlyStopping, this.checkpointBuilder, this.config);
        }
    }

    public void train(DataLoader trainDataset, DataLoader testDataset, long epochs) {
        LOG.info("Training model...");

        int epoch = 0;
        while (epoch < epochs) {
            epoch++;

            trainDataset.reset();
            testDataset.reset();

            // Train loop
            while(trainDataset.hasNext()) {
                DataLoader.Batch trainBatch = trainDataset.next();
                this.trainStep(trainBatch.x(), trainBatch.y());
            }

            // Evaluation
            float totalWeightedMSE  = 0;
            float totalWeightedLoss = 0;
            int totalSamples = 0;
            while(testDataset.hasNext()) {
                DataLoader.Batch batch = testDataset.next();
                Tensor batchPrediction = this.predict(batch.x());

                int batchSize = batch.y().getShape()[0];
                float batchMSE = this.mse.compute(batch.y(), batchPrediction);
                float batchLoss = this.lossFunction.computeLoss(batch.y(), batchPrediction);

                totalWeightedMSE += batchMSE * batchSize;
                totalWeightedLoss += batchLoss * batchSize;

                totalSamples += batchSize;
            }

            float mse = totalWeightedMSE / totalSamples;
            float loss = totalWeightedLoss / totalSamples;

            LOG.debug("Epoch: {} / {}\nMSE: {}\nLOSS: {}", epoch, epochs, mse, loss);

            boolean isBest = this.earlyStopping.update(loss);
            if (isBest && this.earlyStopping.shouldRestoreBestWeights()) {
                this.modelCheckpoint.onNewBest();
            }
            if (this.earlyStopping.shouldStop()) {
                LOG.info("Early stopping at epoch {}", epoch);
                this.modelCheckpoint.onTrainingEnd();
                break;
            }
        }
        LOG.info("Finished training in: {} epochs", epoch);
    }

    private void trainStep(Tensor xInput,  Tensor yInput) {
        // Forward
        Tensor batchPrediction = this.predict(xInput);

        // Error
        Tensor batchErrors = this.lossFunction.computeDelta(yInput, batchPrediction);

        // Backward
        this.backward(batchErrors);

        // Update
        this.updateLayers();
    }

    public Tensor predict(Tensor input) {
        Tensor output = input;

        for (Layer layer : this.layers) {
            output = layer.forward(output);
        }

        return output;
    }

    public void backward(Tensor errors) {
        Tensor delta = errors;
        for (int i = this.layers.length - 1; i >= 0; i--) {
            delta = this.layers[i].backward(delta);
        }
    }

    public void updateLayers() {
        for (Layer layer : this.layers) {
            layer.update(this.optimizer);
        }
    }

    public void save(Path zipPath) {
        ModelSerializer.saveModel(this, zipPath);
    }

    public void save(String zipPath) {
        ModelSerializer.saveModel(this, zipPath);
    }

    public static Model load(String zipPath) {
        return ModelSerializer.loadModel(zipPath);
    }

    public static Model load(Path zipPath) {
        return ModelSerializer.loadModel(zipPath);
    }

    public void copyWeightsFrom(Model model) {
        for (int layerId = 0; layerId < this.layers.length; layerId++) {
            this.layers[layerId].copyParametersFrom(model.layers[layerId]);
        }
    }
}
