package io.h3ca.forgemind.core.utils;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;

import static java.lang.foreign.ValueLayout.*;

public final class NativeMemoryUtils {

    private NativeMemoryUtils() {}

    public static MemorySegment fromIntArray(Arena arena, int[] javaArr) {
        MemorySegment nativeArr = arena.allocate(JAVA_INT, javaArr.length);
        nativeArr.copyFrom(MemorySegment.ofArray(javaArr));
        return nativeArr;
    }

    public static MemorySegment fromFloatArray(Arena arena, float[] javaArr) {
        MemorySegment nativeArr = arena.allocate(JAVA_FLOAT, javaArr.length);
        nativeArr.copyFrom(MemorySegment.ofArray(javaArr));
        return nativeArr;
    }
}
