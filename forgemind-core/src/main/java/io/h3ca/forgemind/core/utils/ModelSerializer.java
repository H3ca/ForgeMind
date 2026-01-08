package io.h3ca.forgemind.core.utils;

import io.h3ca.forgemind.core.api.Model;
import io.h3ca.forgemind.core.api.Tensor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import tools.jackson.databind.JsonNode;
import tools.jackson.databind.ObjectMapper;
import tools.jackson.databind.node.ObjectNode;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;
import java.util.zip.ZipOutputStream;

public class ModelSerializer {

    private static final Logger LOG = LoggerFactory.getLogger(ModelSerializer.class);
    private static final String WARN_EXTENSION = "Recommended extension is {}";

    private static final String MODEL_EXTENSION = ".fmind";

    private static final ObjectMapper MAPPER = new ObjectMapper();

    private static final int MAGIC_NUMBER = 0x464D4E44;
    private static final int VERSION = 1;

    private ModelSerializer() {}

    public static void saveModel(Model model, Path zipPath) {
        if (!zipPath.toString().endsWith(MODEL_EXTENSION)) {
            LOG.warn(WARN_EXTENSION, MODEL_EXTENSION);
        }
        LOG.debug("Saving model to {}", zipPath);

        JsonNode modelNode = MAPPER.valueToTree(model);
        JsonNode layersNode = modelNode.get("layers");

        int tensorCount = 0;
        for (JsonNode layerNode : layersNode) {
            if (layerNode.has("weights")) tensorCount++;
            if (layerNode.has("biases")) tensorCount++;
        }

        LOG.debug("Model format version {}, tensors {}", VERSION, tensorCount);
        try (ZipOutputStream zos = new ZipOutputStream(Files.newOutputStream(zipPath));
             DataOutputStream out = new DataOutputStream(zos)) {
            // CONFIG
            zos.putNextEntry(new ZipEntry("model.json"));

            ObjectNode configCopy = ((ObjectNode) modelNode.deepCopy());
            configCopy.remove("layers");

            ByteArrayOutputStream buffer = new ByteArrayOutputStream();
            MAPPER.writeValue(buffer, configCopy);

            zos.write(buffer.toByteArray());
            zos.closeEntry();

            // WEIGHTS
            ZipEntry weightsEntry = new ZipEntry("model.weights");
            zos.putNextEntry(weightsEntry);


            out.writeInt(MAGIC_NUMBER);
            out.writeInt(VERSION);
            out.writeInt(tensorCount);

            int layerCounter = 0;
            for (JsonNode layerNode : layersNode) {
                layerCounter++;
                serializeProperty(out, layerNode, "weights", layerCounter);
                serializeProperty(out, layerNode, "biases", layerCounter);
            }
            out.flush();
            zos.closeEntry();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
    public static void saveModel(Model model, String zipPath) {
        saveModel(model, Path.of(zipPath));
    }

    public static Model loadModel(Path zipPath) {
        if (!zipPath.toString().endsWith(MODEL_EXTENSION)) {
            LOG.warn(WARN_EXTENSION, MODEL_EXTENSION);
        }
        LOG.debug("Loading model from {}", zipPath);

        ObjectNode modelNode = null;
        JsonNode layersNode = null;

        try (ZipInputStream zis = new ZipInputStream(Files.newInputStream(zipPath))) {
            ZipEntry entry;
            while ((entry = zis.getNextEntry()) != null) {
                if (entry.getName().equals("model.json")) {
                    ByteArrayOutputStream buffer = new ByteArrayOutputStream();
                    zis.transferTo(buffer);
                    modelNode = (ObjectNode) MAPPER.readTree(new ByteArrayInputStream(buffer.toByteArray()));
                    layersNode = modelNode.get("config").get("layers");
                    if (layersNode == null) throw new RuntimeException("Config missing 'layers'");

                } else if (entry.getName().equals("model.weights")) {
                    ByteArrayOutputStream buffer = new ByteArrayOutputStream();
                    zis.transferTo(buffer);

                    try (DataInputStream in = new DataInputStream(new ByteArrayInputStream(buffer.toByteArray()))) {
                        int magic = in.readInt();
                        if (magic != MAGIC_NUMBER)
                            throw new RuntimeException(String.format(
                                    "Invalid ForgeMind model file. Expected 0x%08X, got 0x%08X",
                                    MAGIC_NUMBER, magic
                            ));
                        int version = in.readInt();
                        if (version != VERSION)
                            throw new RuntimeException("Unsupported ForgeMind model format version: " + version + ".");
                        int numTensors = in.readInt();

                        for (int tensorId = 0; tensorId < numTensors; tensorId++) {
                            String[] split = in.readUTF().split("/");
                            String layerName = split[0];
                            String tensorType = split[1];

                            int layerId = Integer.parseInt(layerName.replaceAll("\\D+", "")) - 1;

                            if (layersNode == null) throw new RuntimeException("Config missing layers.");
                            ObjectNode layerObject = (ObjectNode) layersNode.get(layerId);

                            Tensor tensor = deserializeTensor(in);

                            layerObject.putPOJO(tensorType, MAPPER.valueToTree(tensor));
                        }
                    }
                }
                zis.closeEntry();
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        return MAPPER.convertValue(modelNode, Model.class);
    }
    public static Model loadModel(String zipPath) {
        return loadModel(Path.of(zipPath));
    }

    private static void serializeProperty(DataOutputStream out, JsonNode layerNode, String tensorProperty, int layerIndex) throws IOException {

        if (!layerNode.has(tensorProperty)) return;

        String layerType = layerNode.get("type").asString().toLowerCase();
        String tensorName = layerType + layerIndex + "/" + tensorProperty;

        JsonNode propertyNode = layerNode.get(tensorProperty);
        float[] data = MAPPER.convertValue(propertyNode.get("data"), float[].class);
        int[] dims = MAPPER.convertValue(propertyNode.get("shape"), int[].class);

        // Name
        out.writeUTF(tensorName);
        // Rank
        out.writeInt(dims.length);
        // Shape
        for (int shape : dims) out.writeInt(shape);
        // Data
        for (float datum : data) out.writeFloat(datum);
    }
    private static Tensor deserializeTensor(DataInputStream in) throws IOException {
        int rank = in.readInt();
        int[] shape = new int[rank];

        for (int i = 0; i < rank; i++) shape[i] = in.readInt();

        int dataLength = Arrays.stream(shape).reduce(1, Math::multiplyExact);
        float[] data = new float[dataLength];

        for (int i = 0; i < dataLength; i++) data[i] = in.readFloat();


        return new Tensor(data, shape);
    }
}
