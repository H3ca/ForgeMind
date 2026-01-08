package io.h3ca.forgemind.core.loss;

import com.fasterxml.jackson.annotation.JsonTypeName;
import io.h3ca.forgemind.core.api.Tensor;

import java.util.Arrays;

@JsonTypeName("L1Loss")
public class L1Loss implements LossFunction {
    public static L1Loss instance;

    private Tensor deltaTensor;

    private L1Loss() {}

    public static L1Loss get() {
        if (instance == null) instance = new L1Loss();
        return instance;
    }

    @Override
    public float computeLoss(Tensor yBatch, Tensor predictionBatch) {
        int numSamples = predictionBatch.getShape()[0];
        int numClasses = predictionBatch.getShape()[1];

        float[] labelsData = yBatch.getData();
        float[] predictionsData = predictionBatch.getData();

        float loss = 0.0f;

        for (int sampleId = 0; sampleId < numSamples; sampleId++) {
            int baseOffset = sampleId * numClasses;

            for (int classId = 0; classId < numClasses; classId++) {
                int offset = baseOffset + classId;
                loss += Math.abs(predictionsData[offset] - labelsData[offset]);
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

                delta[offset] = Math.signum(predictionsData[offset] - labelsData[offset]);
            }
        }

        return this.deltaTensor;
    }
}
