package io.h3ca.forgemind.core.api;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import io.h3ca.forgemind.core.utils.NativeMemoryUtils;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.Arrays;

import static java.lang.foreign.ValueLayout.JAVA_FLOAT;

public class Tensor {

    private final float[] data;
    private final int[] shape;
    private final int[] strides;

    @JsonCreator
    public Tensor(@JsonProperty("data") float[] data, @JsonProperty("shape") int[] shape) {
        int expected = Arrays.stream(shape).reduce(1, Math::multiplyExact);
        if (data.length != expected) {
            throw new IllegalArgumentException(
                    "Data length " + data.length + " does not match shape " + Arrays.toString(shape) + "."
            );
        }

        this.shape = Arrays.copyOf(shape, shape.length);
        this.data = data;
        this.strides = new int[shape.length];
        this.strides[this.shape.length - 1] = 1;
        for (int i = shape.length - 2; i >= 0; i--) {
            this.strides[i] = this.strides[i + 1] * this.shape[i + 1];
        }
    }

    public Tensor(int[] shape) {
        int length = Arrays.stream(shape).reduce(1, Math::multiplyExact);
        this(new float[length], shape);
    }

    public float[] getData() {
        return this.data;
    }

    public void updateData(float[] data) {
        if (data.length != this.data.length) throw new IllegalArgumentException("Mismatched tensor size.");
        System.arraycopy(data, 0, this.data, 0, data.length);
    }

    public int[] getShape() {
        return this.shape;
    }

    public int[] getStrides() {
        return this.strides;
    }

    public void reset() {
        Arrays.fill(this.data, 0.0f);
    }

    public int rank() {
        return this.shape.length;
    }

    public int size() {
        return this.data.length;
    }

    public int dim(int i) {
        return this.shape[i];
    }

    public int getOffset(int... indices) {
        int offset = 0;
        for (int i = 0; i < indices.length; i++) {
            offset += indices[i] * this.strides[i];
        }
        return offset;
    }

    public Tensor slice(int index) {
        int[] shape = Arrays.copyOf(this.shape, this.shape.length);
        shape[0] = 1;
        Tensor slice = new Tensor(shape);

        float[] sliceData = slice.getData();
        int offset = this.getOffset(index);
        System.arraycopy(this.data, offset, sliceData, 0, sliceData.length);

        return slice;
    }

    /**
     * Serializes this Tensor into native memory allocated in the given Arena.
     * The returned MemorySegment is only valid for the Arena's lifetime.
     */
    public MemorySegment toNative(Arena arena) {

        MemorySegment nativeTensor = io.h3ca.forgemind.core.internal.bindings.Tensor.allocate(arena);

        MemorySegment nativeData = NativeMemoryUtils.fromFloatArray(arena, this.data);
        MemorySegment nativeShape = NativeMemoryUtils.fromIntArray(arena, this.shape);
        MemorySegment nativeStrides = NativeMemoryUtils.fromIntArray(arena, this.strides);

        io.h3ca.forgemind.core.internal.bindings.Tensor.data(nativeTensor, nativeData);
        io.h3ca.forgemind.core.internal.bindings.Tensor.shape(nativeTensor, nativeShape);
        io.h3ca.forgemind.core.internal.bindings.Tensor.strides(nativeTensor, nativeStrides);

        return nativeTensor;
    }

    /**
     * Copies native tensor data back into this Tensor.
     * Shape and strides are assumed to be unchanged.
     */
    public void fromNative(MemorySegment nativeTensor) {
        MemorySegment nativeData = io.h3ca.forgemind.core.internal.bindings.Tensor.data(nativeTensor)
                .reinterpret((long) this.data.length * JAVA_FLOAT.byteSize());
        MemorySegment.ofArray(this.data).copyFrom(nativeData);
    }

    @Override
    public Tensor clone() {
        float[] data = Arrays.copyOf(this.data, this.data.length);
        int[] shape = Arrays.copyOf(this.shape, this.shape.length);
        return new Tensor(data, shape);
    }
}
