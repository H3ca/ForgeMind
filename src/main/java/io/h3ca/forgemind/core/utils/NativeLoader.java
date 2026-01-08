package io.h3ca.forgemind.core.utils;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;

public class NativeLoader {

    private static boolean loaded = false;

    private NativeLoader() {}

    public static void load() {
        if (loaded) return;

        String os = System.getProperty("os.name").toLowerCase();
        String arch = System.getProperty("os.arch").toLowerCase();

        String libFolder;
        String libName;

        if (os.contains("win")) {
            libFolder = "native/windows-x64";
            libName = "forgemind.dll";
        } else if (os.contains("linux")) {
            libFolder = "native/linux-x64";
            libName = "libforgemind.so";
        } else if (os.contains("mac") || os.contains("darwin")) {
            libFolder = "native/mac-x64";
            libName = "libforgemind.dylib";
        } else {
            throw new UnsupportedOperationException("Unsupported OS: " + os + ". ForgeMind native library not available.");
        }

        try {
            // Try to load from system path first
            File libFile = new File(libFolder, libName);
            if (libFile.exists()) {
                System.load(libFile.getAbsolutePath());
                loaded = true;
                return;
            }

            // Otherwise, extract from resources in the JAR
            String resourcePath = "/" + libFolder + "/" + libName;
            InputStream in = NativeLoader.class.getResourceAsStream(resourcePath);
            if (in == null) {
                throw new RuntimeException("ForgeMind native library not found in resources: " + resourcePath);
            }

            File tempFile = Files.createTempFile("forgemind-", "-" + libName).toFile();
            tempFile.deleteOnExit();

            Files.copy(in, tempFile.toPath(), StandardCopyOption.REPLACE_EXISTING);
            System.load(tempFile.getAbsolutePath());

            loaded = true;
        } catch (UnsatisfiedLinkError | IOException e) {
            throw new RuntimeException(
                    "Failed to load ForgeMind native library (" + libName + "). " +
                            "Make sure the correct native library for your OS/architecture is bundled or available.",
                    e
            );
        }
    }
}
