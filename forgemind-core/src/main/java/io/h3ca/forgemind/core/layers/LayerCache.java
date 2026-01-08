package io.h3ca.forgemind.core.layers;

import io.h3ca.forgemind.core.api.Tensor;

public class LayerCache {

    private Tensor input;
    private Tensor output;
    private Tensor deltaOutput;
    private Tensor deltaInput;

    private Tensor weightsGradient;
    private Tensor biasesGradient;

    public LayerCache() {}

    public Tensor getInput() {
        return this.input;
    }

    public void setInput(Tensor input) {
        this.ensureInput(input.getShape());
        this.input.updateData(input.getData());
    }

    public Tensor getOutput() {
        return this.output;
    }

    public Tensor getDeltaOutput() {
        return this.deltaOutput;
    }

    public void setDeltaOutput(Tensor deltaOutput) {
        this.ensureDeltaOutput(deltaOutput.getShape());
        this.deltaOutput.updateData(deltaOutput.getData());
    }

    public Tensor getInputDelta() {
        return this.deltaInput;
    }

    public Tensor getWeightsGradient() {
        return this.weightsGradient;
    }

    public Tensor getBiasesGradient() {
        return this.biasesGradient;
    }

    public void ensureInput(int[] shape) {
        this.ensureInput(shape, null);
    }

    public void ensureInput(int[] shape, float[] dataBuffer) {
        if (this.input == null || !sameShape(this.input.getShape(), shape)) {
            this.input = (dataBuffer != null) ? new Tensor(dataBuffer, shape) : new Tensor(shape);
        }
    }

    public void ensureOutput(int[] shape) {
        this.ensureOutput(shape, null);
    }

    public void ensureOutput(int[] shape, float[] dataBuffer) {
        if (this.output == null || !sameShape(this.output.getShape(), shape)) {
            this.output = (dataBuffer != null) ? new Tensor(dataBuffer, shape) : new Tensor(shape);
        }
    }

    public void ensureDeltaOutput(int[] shape) {
        if (this.deltaOutput == null || !sameShape(this.deltaOutput.getShape(), shape)) {
            this.deltaOutput = new Tensor(shape);
        }
    }

    public void ensureInputDelta(int[] shape, float[] dataBuffer) {
        if (this.deltaInput == null || !sameShape(this.deltaInput.getShape(), shape)) {
            this.deltaInput = (dataBuffer != null) ? new Tensor(dataBuffer, shape) : new Tensor(shape);
        }
    }

    public void ensureInputDelta(int[] shape) {
        this.ensureInputDelta(shape, null);
    }

    public void ensureWeightsGradients(int[] shape) {
        if (this.weightsGradient == null || !sameShape(this.weightsGradient.getShape(), shape)) {
            this.weightsGradient = new Tensor(shape);
        }
    }

    public void ensureBiasesGradients(int[] shape) {
        if (this.biasesGradient == null || !sameShape(this.biasesGradient.getShape(), shape)) {
            this.biasesGradient = new Tensor(shape);
        }
    }

    private static boolean sameShape(int[] a, int[] b) {
        if (a.length != b.length) return false;
        for (int i = 0; i < a.length; i++) {
            if (a[i] != b[i]) return false;
        }
        return true;
    }

    public void reset() {
        this.reset(this.output);
        this.reset(this.weightsGradient);
        this.reset(this.biasesGradient);
    }

    public void reset(Tensor tensor) {
        if (tensor != null) {
            tensor.reset();
        }
    }

}
