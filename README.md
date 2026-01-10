# ForgeMind

## Overview

ForgeMind is a lightweight Java machine learning framework.  
The project currently ships as a **single Maven module**, but it is designed to be split into multiple modules (`core`, `datasets`, `examples`) as the API stabilizes.

---

## Requirements

- **Java 25 (JDK 25)** is required to build and run ForgeMind.
- Platform: **Windows (x64)**. Support for Linux and macOS is planned.

---

## Native Backend Support

ForgeMind uses **native libraries** for performance-critical operations.  
If you attempt to run ForgeMind on an unsupported platform, the framework will fail fast with a clear error message.

---

## Running ForgeMind with Native Libraries

### JVM Option

To allow the framework to access native code, your JVM must enable native access for the `core` module:

```bash
--enable-native-access=ALL-UNNAMED