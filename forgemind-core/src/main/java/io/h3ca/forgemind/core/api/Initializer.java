package io.h3ca.forgemind.core.api;

import com.fasterxml.jackson.annotation.*;

import java.util.SplittableRandom;

@JsonTypeInfo(
        use = JsonTypeInfo.Id.NAME,
        include = JsonTypeInfo.As.PROPERTY,
        property = "type"
)
@JsonSubTypes({
        @JsonSubTypes.Type(value = Initializer.Constant.class, name = "Constant"),
        @JsonSubTypes.Type(value = Initializer.Ones.class, name = "Ones"),
        @JsonSubTypes.Type(value = Initializer.Zeros.class, name = "Zeros"),
        @JsonSubTypes.Type(value = Initializer.GlorotUniform.class, name = "GlorotUniform"),
        @JsonSubTypes.Type(value = Initializer.GlorotGaussian.class, name = "GlorotGaussian"),
        @JsonSubTypes.Type(value = Initializer.RandomUniform.class, name = "RandomUniform"),
        @JsonSubTypes.Type(value = Initializer.RandomGaussian.class, name = "RandomGaussian")
})
public interface Initializer {
    void initialize(Tensor weights);

    @JsonTypeName("Constant")
    public static class Constant implements Initializer {

        @JsonProperty("scalar")
        private final float scalar;

        @JsonCreator
        public Constant(@JsonProperty("scalar") float scalar) {
            this.scalar = scalar;
        }

        public Constant() {
            this(0.0f);
        }

        @Override
        public void initialize(Tensor weights) {
            float[] data = weights.getData();
            int size = weights.size();

            for (int i = 0; i < size; i++) {
                data[i] = this.scalar;
            }
        }
    }

    @JsonTypeName("Ones")
    public static class Ones implements Initializer {

        @Override
        public void initialize(Tensor weights) {
            float[] data = weights.getData();
            int size = weights.size();

            for (int i = 0; i < size; i++) {
                data[i] = 1;
            }
        }
    }

    @JsonTypeName("Zeros")
    public static class Zeros implements Initializer {

        @Override
        public void initialize(Tensor weights) {
            float[] data = weights.getData();
            int size = weights.size();

            for (int i = 0; i < size; i++) {
                data[i] = 0;
            }
        }
    }

    @JsonTypeName("GlorotUniform")
    public static class GlorotUniform implements Initializer {

        private final SplittableRandom random;
        @JsonProperty("seed")
        private final long seed;

        @JsonCreator
        public GlorotUniform(@JsonProperty("seed") long seed) {
            this.seed = seed;
            this.random = new SplittableRandom(seed);
        }

        public GlorotUniform() {
            this(System.currentTimeMillis());
        }

        @Override
        public void initialize(Tensor weights) {
            float[] data = weights.getData();
            int size = weights.size();
            int rank = weights.rank();
            int[] shape = weights.getShape();

            float limit = (float) Math.sqrt((float) 6 / (shape[rank - 2] + shape[rank - 1]));

            for (int i = 0; i < size; i++) {
                data[i] = this.random.nextFloat(-limit, limit);
            }
        }
    }

    @JsonTypeName("GlorotGaussian")
    public static class GlorotGaussian implements Initializer {

        private final SplittableRandom random;
        @JsonProperty("seed")
        private final long seed;

        @JsonCreator
        public GlorotGaussian(@JsonProperty("seed") long seed) {
            this.seed = seed;
            this.random = new SplittableRandom(seed);
        }

        public GlorotGaussian() {
            this(System.currentTimeMillis());
        }

        @Override
        public void initialize(Tensor weights) {
            float[] data = weights.getData();
            int size = weights.size();
            int rank = weights.rank();
            int[] shape = weights.getShape();

            float stddev = (float) Math.sqrt((float) 2 / ((shape[rank - 2] + shape[rank - 1])));

            for (int i = 0; i < size; i++) {
                data[i] = (float) this.random.nextGaussian(0, stddev);
            }
        }
    }

    @JsonTypeName("RandomUniform")
    public static class RandomUniform implements Initializer {

        private final SplittableRandom random;
        @JsonProperty("seed")
        private final long seed;
        @JsonProperty("min")
        private final float min;
        @JsonProperty("max")
        private final float max;

        @JsonCreator
        public RandomUniform(
                @JsonProperty("seed") long seed,
                @JsonProperty("min") float min,
                @JsonProperty("max") float max)
        {
            this.seed = seed;
            this.random = new SplittableRandom(seed);
            this.min = min;
            this.max = max;
        }

        public RandomUniform(float min, float max) {
            this(System.currentTimeMillis(), min, max);
        }

        @Override
        public void initialize(Tensor weights) {
            float[] data = weights.getData();
            int size = weights.size();

            for (int i = 0; i < size; i++) {
                data[i] = this.random.nextFloat(this.min, this.max);
            }
        }
    }

    @JsonTypeName("RandomGaussian")
    public static class RandomGaussian implements Initializer {

        private final SplittableRandom random;
        @JsonProperty("seed")
        private final long seed;
        @JsonProperty("mean")
        private final float mean;
        @JsonProperty("stddev")
        private final float stddev;

        @JsonCreator
        public RandomGaussian(
                @JsonProperty("seed") long seed,
                @JsonProperty("mean") float mean,
                @JsonProperty("stddev") float stddev)
        {
            this.seed = seed;
            this.random = new SplittableRandom(seed);
            this.mean = mean;
            this.stddev = stddev;
        }

        public RandomGaussian(float mean, float stddev) {
            this(System.currentTimeMillis(), mean, stddev);
        }

        @Override
        public void initialize(Tensor weights) {
            float[] data = weights.getData();
            int size = weights.size();

            for (int i = 0; i < size; i++) {
                data[i] = (float) this.random.nextGaussian(this.mean, this.stddev);
            }
        }
    }
}
