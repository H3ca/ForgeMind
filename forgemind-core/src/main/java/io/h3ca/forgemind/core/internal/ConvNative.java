package io.h3ca.forgemind.core.internal;

import io.h3ca.forgemind.core.internal.bindings.forgemind_h;
import io.h3ca.forgemind.core.utils.NativeLoader;

import java.lang.foreign.MemorySegment;

public class ConvNative {
    static {
        NativeLoader.load();
    }

    private ConvNative() {}

    public static void computeGradients(
            MemorySegment padding, MemorySegment delta, MemorySegment input,
            MemorySegment biasGrad, MemorySegment weightGrad) {

        forgemind_h.computeGradients(padding, delta, input, biasGrad, weightGrad);
    }
}
