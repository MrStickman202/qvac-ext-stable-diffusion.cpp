package io.tether.sdnpu

/** sd.cpp through the JNI bridge in src/main/cpp/sd_jni.cpp. Calls block; run them off the main thread. */
object SdNative {
    init {
        System.loadLibrary("sdjni")
    }

    /**
     * Loads a generation context and returns its handle, or 0 on failure (see [lastError]).
     * Empty [qnnUnet] / [qnnTae] run that part with ggml on the CPU instead of the NPU.
     */
    @JvmStatic external fun load(libDir: String, model: String, qnnUnet: String, qnnTae: String, threads: Int): Long

    /** Text-to-image; returns width * height * 3 RGB bytes, or null on failure (see [lastError]). */
    @JvmStatic external fun generate(
        handle: Long, prompt: String, negative: String, seed: Long, steps: Int, cfg: Float, width: Int, height: Int,
    ): ByteArray?

    /** Text encoder, UNet and decoder milliseconds of the last [generate]. */
    @JvmStatic external fun timings(): DoubleArray

    @JvmStatic external fun lastError(): String

    @JvmStatic external fun release(handle: Long)
}
