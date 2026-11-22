package io.tether.sdnpu

import android.app.DownloadManager
import android.content.Context
import android.net.Uri
import java.io.File
import java.security.MessageDigest

/**
 * A model file the app needs: where it goes (relative to the models folder), its exact size and SHA-256, and the
 * URLs it is downloaded from. Several URLs are parts that are joined in order (GitHub serves files up to 100 MB).
 */
data class ModelFile(val path: String, val bytes: Long, val sha256: String, val urls: List<String>)

/**
 * SDXS-512-DreamShaper. sd.cpp loads it from the diffusers folder layout, downloaded from Hugging Face at a pinned
 * revision; the UNet and tiny decoder for the NPU are QNN context binaries compiled for the Snapdragon 8 Elite
 * (SM8750, Hexagon v79) by script/qnn/export_artifacts.py with QAIRT 2.40, the version whose libraries the APK
 * carries. They come from the sdnpu-models branch of the fork, pinned to a commit.
 *
 * Everything lives in the app's external files folder (Android/data/io.tether.sdnpu/files/models), so the files
 * can also be copied there over USB instead of downloaded.
 */
object Models {
    const val NPU_SOC = "SM8750"
    private const val HF = "https://huggingface.co/IDKiro/sdxs-512-dreamshaper/resolve/76f720262bb051da75666b22c902a78c8e16c763/"
    private const val GH =
        "https://raw.githubusercontent.com/MrStickman202/qvac-ext-stable-diffusion.cpp/8f84fd331f9f6b5d81fc2aaec9b5fb4ffe25463a/"

    /** The folder passed to sd.cpp as the model (-m). */
    const val SDXS_DIR = "sdxs"

    val textEncoder = ModelFile("sdxs/text_encoder/model.safetensors", 492_265_168L,
        "5cd76eb3ed326974cfd66795384e890742dfdb5a80f5f4e4cb1547d51ec75d86", listOf(HF + "text_encoder/model.safetensors"))
    val unetWeights = ModelFile("sdxs/unet/diffusion_pytorch_model.safetensors", 1_263_018_648L,
        "a9cb7932239b22d318f20d80c0062be5f06c9e96b620d3719db49b31f23927cf",
        listOf(HF + "unet/diffusion_pytorch_model.safetensors"))
    val vae = ModelFile("sdxs/vae/diffusion_pytorch_model.safetensors", 9_793_292L,
        "d7956d561b1efbd861ad9b03fd8f01510f9e87eddc07bdfd20837009433f6ee5",
        listOf(HF + "vae/diffusion_pytorch_model.safetensors"))
    val unet = ModelFile("sdxs_unet_fp16.bin", 635_904_000L,
        "e340619a262790fff010816f7f13b96bab8a1d9973d1c580029f3e6b6a5be81c",
        (0..6).map { GH + "sdxs_unet_fp16.bin.%03d".format(it) })
    val tae = ModelFile("sdxs_tae_decoder.bin", 3_424_256L,
        "92ab9270e8cff6ea51067460301edef652a3649fd954b0b16315d6d5900a5268", listOf(GH + "sdxs_tae_decoder.bin"))

    val all = listOf(textEncoder, unetWeights, vae, unet, tae)
    val totalBytes = all.sumOf { it.bytes }

    fun dir(context: Context): File = File(context.getExternalFilesDir(null), "models").apply { mkdirs() }

    fun file(context: Context, m: ModelFile) = File(dir(context), m.path)

    fun sdxsDir(context: Context) = File(dir(context), SDXS_DIR)

    /** Present means the right size; the content is checked once, after downloading (see [ModelDownloader]). */
    fun isPresent(context: Context, m: ModelFile) = file(context, m).length() == m.bytes

    fun missing(context: Context) = all.filterNot { isPresent(context, it) }
}

/**
 * Downloads the model files with the system DownloadManager, which keeps going in the background, survives the app
 * being closed and resumes after network drops. Download ids are kept in shared preferences so the app picks up
 * running downloads when it is opened again.
 *
 * Every download goes to a temporary "<file>.dl<part>" name. DownloadManager reserves a file's full size on disk as
 * soon as a download starts, so a file's size says nothing about whether it is finished; only a download that
 * DownloadManager reports as successful is moved (or, for multi-part files, joined) to its final name, and every
 * final file is checked against its SHA-256 once.
 */
class ModelDownloader(private val context: Context) {
    private val dm = context.getSystemService(DownloadManager::class.java)
    // "downloads2": version 0.1 kept ids under "downloads" and wrote straight to the final names. Its files are
    // left alone; finish() checks any that are the right size and deletes the ones that do not match.
    private val prefs = context.getSharedPreferences("downloads2", Context.MODE_PRIVATE)

    data class Progress(val done: Long, val total: Long, val running: Boolean, val failed: String?)

    private fun tempFile(m: ModelFile, i: Int) = File(Models.dir(context), "${m.path}.dl$i")

    private fun tempPath(m: ModelFile, i: Int) = tempFile(m, i).relativeTo(Models.dir(context)).path

    private fun key(m: ModelFile, i: Int) = "${m.path}#$i"

    private fun expectedBytes(m: ModelFile, i: Int): Long =
        if (m.urls.size == 1) m.bytes else minOf(PART_BYTES, m.bytes - i * PART_BYTES)

    fun start() {
        for (m in Models.missing(context)) {
            for ((i, url) in m.urls.withIndex()) {
                val id = prefs.getLong(key(m, i), -1L)
                val status = if (id >= 0) status(id) else -1
                if (status in ACTIVE || status == DownloadManager.STATUS_SUCCESSFUL && tempFile(m, i).exists()) {
                    continue
                }
                if (id >= 0) dm.remove(id)
                tempFile(m, i).apply { parentFile?.mkdirs(); delete() }
                val request = DownloadManager.Request(Uri.parse(url))
                    .setTitle("SD NPU: ${File(m.path).name}" + if (m.urls.size > 1) " (${i + 1}/${m.urls.size})" else "")
                    .setNotificationVisibility(DownloadManager.Request.VISIBILITY_VISIBLE)
                    .setDestinationInExternalFilesDir(context, "models", tempPath(m, i))
                    .setAllowedOverMetered(true)
                prefs.edit().putLong(key(m, i), dm.enqueue(request)).apply()
            }
        }
    }

    fun cancel() {
        for (m in Models.all) {
            for (i in m.urls.indices) {
                val id = prefs.getLong(key(m, i), -1L)
                if (id >= 0) dm.remove(id)  // also deletes the partial file
                prefs.edit().remove(key(m, i)).apply()
                tempFile(m, i).delete()
            }
        }
    }

    fun progress(): Progress {
        var done = 0L
        var running = false
        var failed: String? = null
        for (m in Models.all) {
            if (Models.isPresent(context, m)) {
                done += m.bytes
                continue
            }
            for (i in m.urls.indices) {
                val id = prefs.getLong(key(m, i), -1L)
                if (id < 0) continue
                dm.query(DownloadManager.Query().setFilterById(id)).use { c ->
                    if (!c.moveToFirst()) return@use
                    val status = c.getInt(c.getColumnIndexOrThrow(DownloadManager.COLUMN_STATUS))
                    // Bytes actually received, not the size of the (pre-allocated) file.
                    done += c.getLong(c.getColumnIndexOrThrow(DownloadManager.COLUMN_BYTES_DOWNLOADED_SO_FAR)).coerceAtLeast(0)
                    when (status) {
                        DownloadManager.STATUS_FAILED -> failed = "${File(m.path).name}: download failed (reason " +
                            "${c.getInt(c.getColumnIndexOrThrow(DownloadManager.COLUMN_REASON))}). Tap Download to retry."
                        DownloadManager.STATUS_SUCCESSFUL -> {}
                        else -> running = true
                    }
                }
            }
        }
        return Progress(done, Models.totalBytes, running, failed)
    }

    /**
     * Moves or joins downloads that DownloadManager reports as finished to their final names, and checks the SHA-256
     * of every final file not checked yet (including files copied over USB). Blocking: call it on a background thread.
     * Returns an error message, or null when all is well.
     */
    fun finish(): String? {
        // Version 0.1 downloaded parts as "<file>.partNNN"; they are not used any more.
        Models.dir(context).listFiles()?.filter { Regex(""".*\.part\d{3}$""").matches(it.name) }?.forEach { it.delete() }
        for (m in Models.all) {
            if (!Models.isPresent(context, m)) {
                val ids = m.urls.indices.map { prefs.getLong(key(m, it), -1L) }
                val parts = m.urls.indices.map { tempFile(m, it) }
                val complete = ids.all { it >= 0 && status(it) == DownloadManager.STATUS_SUCCESSFUL } &&
                    parts.withIndex().all { (i, f) -> f.length() == expectedBytes(m, i) }
                if (!complete) continue
                val out = Models.file(context, m)
                out.parentFile?.mkdirs()
                out.delete()
                if (parts.size == 1) {
                    if (!parts[0].renameTo(out)) return "Could not move ${File(m.path).name} into place."
                } else {
                    val tmp = File(out.path + ".joining")
                    tmp.outputStream().use { o -> parts.forEach { p -> p.inputStream().use { it.copyTo(o, 1 shl 20) } } }
                    if (!tmp.renameTo(out)) return "Could not move ${File(m.path).name} into place."
                    parts.forEach { it.delete() }
                }
                prefs.edit().remove("verified:${m.path}").apply()
                // The downloads' records point at files that no longer exist; drop them so they are not reused.
                ids.forEach { dm.remove(it) }
                m.urls.indices.forEach { prefs.edit().remove(key(m, it)).apply() }
            }
            if (!Models.isPresent(context, m) || prefs.getBoolean("verified:${m.path}", false)) continue
            if (sha256(Models.file(context, m)) != m.sha256) {
                Models.file(context, m).delete()
                return "${File(m.path).name} did not match its checksum and has been deleted. Tap Download to fetch it again."
            }
            prefs.edit().putBoolean("verified:${m.path}", true).apply()
        }
        return null
    }

    private fun sha256(f: File): String {
        val md = MessageDigest.getInstance("SHA-256")
        f.inputStream().use { input ->
            val buf = ByteArray(1 shl 20)
            while (true) {
                val n = input.read(buf)
                if (n < 0) break
                md.update(buf, 0, n)
            }
        }
        return md.digest().joinToString("") { "%02x".format(it) }
    }

    private fun status(id: Long): Int =
        dm.query(DownloadManager.Query().setFilterById(id)).use { c ->
            if (c.moveToFirst()) c.getInt(c.getColumnIndexOrThrow(DownloadManager.COLUMN_STATUS)) else -1
        }

    companion object {
        /** Size of every part but the last of a multi-part file (see the sdnpu-models branch). */
        const val PART_BYTES = 95_000_000L
        private val ACTIVE = listOf(DownloadManager.STATUS_RUNNING, DownloadManager.STATUS_PENDING, DownloadManager.STATUS_PAUSED)
    }
}
