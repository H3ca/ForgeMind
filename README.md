# ForgeMind

## Project Structure

ForgeMind currently ships as a single Maven module.  
The project is designed to be split into multiple modules (core, datasets, examples) as the API stabilizes.

---

## Requirements

- Java 25 (JDK 25) is required to build and run ForgeMind.

---

## Native Backend Support

ForgeMind currently ships with a native backend for **Windows (x64)**.

Support for Linux and macOS is planned but not yet available.

If you attempt to run ForgeMind on an unsupported platform, the framework will fail fast with a clear error message.

## Running ForgeMind with Native Libraries

ForgeMind uses native libraries for performance-critical operations.  
To run the framework, your JVM must allow **native access**, which requires a special flag.

### Required JVM Option

All platforms require:

```text
--enable-native-access=ALL-UNNAMED