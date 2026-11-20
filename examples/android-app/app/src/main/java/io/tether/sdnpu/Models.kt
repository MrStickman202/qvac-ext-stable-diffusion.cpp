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
 * running downloads when it is opened again. Multi-part files are joined when their parts are done, and every
 * downloaded file is checked against its SHA-256.
 */
class ModelDownloader(private val context: Context) {
    private val dm = context.getSystemService(DownloadManager::class.java)
    private val prefs = context.getSharedPreferences("downloads", Context.MODE_PRIVATE)

    data class Progress(val done: Long, val total: Long, val running: Boolean, val failed: String?)

    private fun partFile(m: ModelFile, i: Int) =
        if (m.urls.size == 1) Models.file(context, m) else File(Models.dir(context), "${m.path}.part%03d".format(i))

    private fun partPath(m: ModelFile, i: Int) = partFile(m, i).relativeTo(Models.dir(context)).path

    private fun key(m: ModelFile, i: Int) = "${m.path}#$i"

    fun start() {
        for (m in Models.missing(context)) {
            for ((i, url) in m.urls.withIndex()) {
                val id = prefs.getLong(key(m, i), -1L)
                val status = if (id >= 0) status(id) else -1
                if (status in ACTIVE || status == DownloadManager.STATUS_SUCCESSFUL && partFile(m, i).exists()) {
                    continue
                }
                if (id >= 0) dm.remove(id)
                partFile(m, i).apply { parentFile?.mkdirs(); delete() }
                val request = DownloadManager.Request(Uri.parse(url))
                    .setTitle("SD NPU: ${File(m.path).name}" + if (m.urls.size > 1) " (${i + 1}/${m.urls.size})" else "")
                    .setNotificationVisibility(DownloadManager.Request.VISIBILITY_VISIBLE)
                    .setDestinationInExternalFilesDir(context, "models", partPath(m, i))
                    .setAllowedOverMetered(true)
                prefs.edit().putLong(key(m, i), dm.enqueue(request)).apply()
            }
        }
    }

    fun cancel() {
        for (m in Models.all) {
            for (i in m.urls.indices) {
                val id = prefs.getLong(key(m, i), -1L)
                if (id >= 0) dm.remove(id)
                prefs.edit().remove(key(m, i)).apply()
                if (m.urls.size > 1) partFile(m, i).delete()
            }
            if (!Models.isPresent(context, m)) Models.file(context, m).delete()
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
     * Joins finished multi-part files and checks every newly completed file's SHA-256. Blocking: call it on a
     * background thread once [progress] reports nothing running. Returns an error message, or null when all is well.
     */
    fun finish(): String? {
        for (m in Models.all) {
            if (Models.isPresent(context, m) && prefs.getBoolean("verified:${m.path}", false)) continue
            if (m.urls.size > 1 && !Models.isPresent(context, m)) {
                val parts = m.urls.indices.map { partFile(m, it) }
                if (parts.any { !it.exists() }) continue  // still missing parts; not an error yet
                val out = Models.file(context, m)
                out.parentFile?.mkdirs()
                out.outputStream().use { o -> parts.forEach { p -> p.inputStream().use { it.copyTo(o, 1 shl 20) } } }
                parts.forEach { it.delete() }
            }
            if (!Models.isPresent(context, m)) continue
            if (sha256(Models.file(context, m)) != m.sha256) {
                Models.file(context, m).delete()
                m.urls.indices.forEach { prefs.edit().remove(key(m, it)).apply() }
                return "${File(m.path).name} was corrupted in transit and has been deleted. Tap Download to fetch it again."
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
        private val ACTIVE = listOf(DownloadManager.STATUS_RUNNING, DownloadManager.STATUS_PENDING, DownloadManager.STATUS_PAUSED)
    }
}
