package io.h3ca.forgemind.examples;

import io.h3ca.forgemind.core.api.DataLoader;
import io.h3ca.forgemind.datasets.MnistDataset;
import io.h3ca.forgemind.core.api.Model;
import io.h3ca.forgemind.core.api.Tensor;
import io.h3ca.forgemind.core.callbacks.EarlyStopping;
import io.h3ca.forgemind.core.callbacks.ModelCheckpoint;
import io.h3ca.forgemind.core.layers.Conv;
import io.h3ca.forgemind.core.layers.Dense;
import io.h3ca.forgemind.core.layers.Flatten;
import io.h3ca.forgemind.core.layers.activation.Activation;
import io.h3ca.forgemind.core.layers.activation.ActivationFunction;
import io.h3ca.forgemind.core.layers.activation.Softmax;
import io.h3ca.forgemind.core.layers.pooling.Pooling;
import io.h3ca.forgemind.core.loss.CrossEntropy;
import io.h3ca.forgemind.core.loss.LossFunction;
import io.h3ca.forgemind.core.optimizers.Adam;
import io.h3ca.forgemind.core.optimizers.Optimizer;

import java.util.Arrays;

public class ForgeMindExample {

    public void run() {
        runMNISTExperiment();
    }

    private void runMNISTExperiment() {
        System.out.println("=================================================================");
        System.out.println("Running MNIST Experiment");
        System.out.println("=================================================================");

        MnistDataset trainMnist = new MnistDataset(60_000, true);
        MnistDataset testMnist = new MnistDataset(10_000, false);

        trainMnist.setNormalizationFactor(255);
        testMnist.setNormalizationFactor(255);

        DataLoader trainLoader = new DataLoader(trainMnist, 256);
        DataLoader testLoader = new DataLoader(testMnist, 256);

        LossFunction lossFunction = CrossEntropy.get();
        Optimizer optimizer = new Adam.Builder().build();
        EarlyStopping earlyStopping = new EarlyStopping.Builder()
                .patience(5)
                .restoreBestWeights(true)
                .build();
        ModelCheckpoint.Builder checkpointBuilder = new ModelCheckpoint.Builder();

        Model model = new Model.Builder()
                .lossFunction(lossFunction)
                .optimizer(optimizer)
                .earlyStopping(earlyStopping)
                .checkpointBuilder(checkpointBuilder)
                .config(// CONVOLUTIONAL BLOCK 1
                        new Conv.Builder(32, new int[]{3, 3}),
                        new Activation.Builder(ActivationFunction.ReLu),
                        new Pooling.Builder(2, 2, Pooling.Builder.Type.MAX),

                        // CONVOLUTIONAL BLOCK 2
                        new Conv.Builder(64, new int[]{3, 3}),
                        new Activation.Builder(ActivationFunction.ReLu),
                        new Pooling.Builder(2, 2, Pooling.Builder.Type.MAX),

                        // FLATTEN LAYER
                        new Flatten.Builder(),

                        // FULLY CONNECTED LAYER
                        new Dense.Builder(128),
                        new Activation.Builder(ActivationFunction.ReLu),

                        // OUTPUT LAYER
                        new Dense.Builder(10),
                        new Softmax.Builder(10)
                )
                .build();

        model.train(trainLoader, testLoader, 10);

        model.save("res/models/model.fmind");

        for (int i = 0; i < testMnist.getClasses().length; i++) {
            int classNumber = testMnist.getClasses()[i];

            int imageIndex = 0;

            Tensor labelExample;

            imageFound:
            while(true) {
                labelExample = testMnist.getLabel(imageIndex);

                for (int labelIndex = 0; labelIndex < labelExample.size(); labelIndex++) {
                    int labelNumber = Math.toIntExact(Math.round(labelExample.getData()[labelIndex]));
                    if (labelNumber != 1) continue;
                    if (labelIndex == classNumber) break imageFound;
                    break;
                }
                imageIndex++;
            }

            Tensor imageExample = testMnist.getSample(imageIndex);
            Tensor prediction = model.predict(imageExample);

            float[] labelData = labelExample.getData();
            float[] predictionData = prediction.getData();

            System.out.println("Label:      " + Arrays.toString(labelData));
            System.out.println("Prediction: " + Arrays.toString(predictionData));
        }
    }
}
