package io.h3ca.forgemind.core.loss;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.JsonTypeName;
import io.h3ca.forgemind.core.api.Tensor;

import java.util.Arrays;

@JsonTypeName("HuberLoss")
public class HuberLoss implements LossFunction {

    @JsonProperty("delta")
    private final float delta;

    private Tensor deltaTensor;

    @JsonCreator
    public HuberLoss(@JsonProperty("delta") float delta) {
        if (delta <= 0) throw new IllegalArgumentException("delta must be positive.");
        this.delta = delta;
    }

    @Override
    public float computeLoss(Tensor yBatch, Tensor predictionBatch) {
        int numSamples = predictionBatch.getShape()[0];
        int numClasses = predictionBatch.getShape()[1];

        float[] labelsData = yBatch.getData();
        float[] predictionsData = predictionBatch.getData();

        float loss = 0;
        for (int sampleId = 0; sampleId < numSamples; sampleId++) {
            int baseOffset = sampleId * numClasses;

            for (int classId = 0; classId < numClasses; classId++) {
                int offset = baseOffset + classId;
                float error = predictionsData[offset] - labelsData[offset];
                loss += Math.abs(error) <= this.delta ?
                        (float) (0.5 * error * error) :
                        this.delta * (float)(Math.abs(error) - 0.5 * this.delta);
            }
        }

        return loss / numSamples;
    }

        @Override
        public Tensor computeDelta(Tensor yBatch, Tensor predictionBatch) {
            int numSamples = predictionBatch.getShape()[0];
            int numClasses = predictionBatch.getShape()[1];

            float[] labelsData = yBatch.getData();
            float[] predictionsData = predictionBatch.getData();

            if (this.deltaTensor == null || !Arrays.equals(this.deltaTensor.getShape(), new int[]{numSamples, numClasses})) {
                this.deltaTensor = new Tensor(new int[]{numSamples, numClasses});
            }

            float[] delta = this.deltaTensor.getData();
            for (int sampleId = 0; sampleId < numSamples; sampleId++) {
                int baseOffset = sampleId * numClasses;

                for (int classId = 0; classId < numClasses; classId++) {
                    int offset = baseOffset + classId;

                    float error = predictionsData[offset] - labelsData[offset];
                    delta[offset] = Math.abs(error) <= this.delta ?
                            error :
                            this.delta * Math.signum(error);
                }
            }
            return this.deltaTensor;
        }
    }
