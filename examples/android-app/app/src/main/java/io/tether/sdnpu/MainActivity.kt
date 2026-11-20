package io.tether.sdnpu

import android.app.Application
import android.content.ContentValues
import android.graphics.Bitmap
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.outlined.AutoAwesome
import androidx.compose.material.icons.outlined.CloudDownload
import androidx.compose.material.icons.outlined.Download
import androidx.compose.material.icons.outlined.Memory
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilledTonalButton
import androidx.compose.material3.Icon
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.SegmentedButton
import androidx.compose.material3.SegmentedButtonDefaults
import androidx.compose.material3.SingleChoiceSegmentedButtonRow
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.dynamicDarkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.compose.viewModel
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.concurrent.Executors
import kotlin.random.Random

enum class Engine(val label: String) { NPU("NPU"), CPU("CPU") }

data class Timings(val engine: Engine, val textMs: Double, val unetMs: Double, val decodeMs: Double, val totalMs: Double)

data class UiState(
    val soc: String = Build.SOC_MODEL,
    val npuSupported: Boolean = Build.SOC_MODEL == Models.NPU_SOC,
    val modelsReady: Boolean = false,
    val download: ModelDownloader.Progress? = null,
    val verifying: Boolean = false,
    val engine: Engine = Engine.NPU,
    val loadedEngine: Engine? = null,
    val loading: Boolean = false,
    val generating: Boolean = false,
    val image: Bitmap? = null,
    val seed: Long = 0,
    val timings: Timings? = null,
    val message: String? = null,
    val error: String? = null,
)

class MainViewModel(app: Application) : AndroidViewModel(app) {
    val state = MutableStateFlow(UiState())
    private val downloader = ModelDownloader(app)
    // sd.cpp contexts are not thread-safe: every native call runs on this one thread.
    private val native = Executors.newSingleThreadExecutor().asCoroutineDispatcher()
    private var handle = 0L

    init {
        state.update { it.copy(engine = if (it.npuSupported) Engine.NPU else Engine.CPU) }
        refreshModels()
    }

    private fun refreshModels() {
        val progress = downloader.progress()
        state.update { it.copy(download = progress) }
        if (progress.running) pollDownloads() else finishModels()
    }

    /** Joins downloaded parts and checks the files (once), then loads the model if everything is there. */
    private fun finishModels() {
        viewModelScope.launch {
            state.update { it.copy(verifying = true) }
            val err = withContext(Dispatchers.IO) { downloader.finish() }
            val ready = Models.missing(getApplication()).isEmpty()
            state.update {
                it.copy(verifying = false, modelsReady = ready, download = if (ready) null else downloader.progress(),
                    error = err ?: it.error)
            }
            if (ready) load(state.value.engine)
        }
    }

    fun download() {
        state.update { it.copy(error = null) }
        downloader.start()
        pollDownloads()
    }

    fun cancelDownload() {
        downloader.cancel()
        state.update { it.copy(download = downloader.progress()) }
    }

    private var polling = false

    private fun pollDownloads() {
        if (polling) return
        polling = true
        viewModelScope.launch {
            while (true) {
                val p = withContext(Dispatchers.IO) { downloader.progress() }
                state.update { it.copy(download = p, error = p.failed) }
                if (!p.running) break
                delay(500)
            }
            polling = false
            finishModels()
        }
    }

    fun setEngine(engine: Engine) {
        if (engine == state.value.engine || state.value.loading || state.value.generating) return
        state.update { it.copy(engine = engine) }
        if (state.value.modelsReady) load(engine)
    }

    private fun load(engine: Engine) {
        val app = getApplication<Application>()
        state.update { it.copy(loading = true, loadedEngine = null, error = null, message = null) }
        viewModelScope.launch {
            val (h, err) = withContext(native) {
                if (handle != 0L) SdNative.release(handle).also { handle = 0L }
                val npu = engine == Engine.NPU
                val h = SdNative.load(
                    app.applicationInfo.nativeLibraryDir,
                    Models.sdxsDir(app).path,
                    if (npu) Models.file(app, Models.unet).path else "",
                    if (npu) Models.file(app, Models.tae).path else "",
                    0,
                )
                h to (if (h == 0L) SdNative.lastError().trim() else "")
            }
            handle = h
            if (h != 0L) {
                state.update { it.copy(loading = false, loadedEngine = engine) }
            } else if (engine == Engine.NPU) {
                // Fall back to the CPU so the app stays usable, and say why.
                state.update { it.copy(loading = false, engine = Engine.CPU, error = "The NPU could not load the model: $err\nSwitched to the CPU.") }
                load(Engine.CPU)
            } else {
                state.update { it.copy(loading = false, error = "Could not load the model: $err") }
            }
        }
    }

    fun generate(prompt: String) {
        val s = state.value
        if (handle == 0L || s.loading || s.generating || prompt.isBlank()) return
        val seed = Random.nextLong(0, Int.MAX_VALUE.toLong())
        val engine = s.loadedEngine ?: return
        state.update { it.copy(generating = true, error = null, message = null, seed = seed) }
        viewModelScope.launch {
            val t0 = System.nanoTime()
            val (rgb, timings, err) = withContext(native) {
                val rgb = SdNative.generate(handle, prompt.trim(), "", seed, 1, 1.0f, SIZE, SIZE)
                Triple(rgb, SdNative.timings(), if (rgb == null) SdNative.lastError().trim() else "")
            }
            val totalMs = (System.nanoTime() - t0) / 1e6
            if (rgb == null) {
                state.update { it.copy(generating = false, error = "Generation failed: $err") }
                return@launch
            }
            val bitmap = withContext(Dispatchers.Default) { toBitmap(rgb) }
            state.update {
                it.copy(
                    generating = false, image = bitmap,
                    timings = Timings(engine, timings[0], timings[1], timings[2], totalMs),
                )
            }
        }
    }

    fun save() {
        val bitmap = state.value.image ?: return
        val app = getApplication<Application>()
        viewModelScope.launch(Dispatchers.IO) {
            val values = ContentValues().apply {
                put(MediaStore.Images.Media.DISPLAY_NAME, "sdnpu_${state.value.seed}.png")
                put(MediaStore.Images.Media.MIME_TYPE, "image/png")
                put(MediaStore.Images.Media.RELATIVE_PATH, "Pictures/SD NPU")
            }
            val uri = app.contentResolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values)
            val ok = uri != null && app.contentResolver.openOutputStream(uri)?.use {
                bitmap.compress(Bitmap.CompressFormat.PNG, 100, it)
            } == true
            state.update { it.copy(message = if (ok) "Saved to Pictures/SD NPU" else "Could not save the image") }
        }
    }

    override fun onCleared() {
        val h = handle
        handle = 0L
        // Free the context on its own thread, after anything still running there.
        native.executor.execute {
            if (h != 0L) SdNative.release(h)
        }
        native.executor.execute { native.close() }
    }

    private fun toBitmap(rgb: ByteArray): Bitmap {
        val pixels = IntArray(SIZE * SIZE) { i ->
            val r = rgb[3 * i].toInt() and 0xff
            val g = rgb[3 * i + 1].toInt() and 0xff
            val b = rgb[3 * i + 2].toInt() and 0xff
            (0xff shl 24) or (r shl 16) or (g shl 8) or b
        }
        return Bitmap.createBitmap(pixels, SIZE, SIZE, Bitmap.Config.ARGB_8888)
    }

    companion object {
        const val SIZE = 512  // SDXS is distilled at 512x512 only
    }
}

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            val context = LocalContext.current
            val colors = if (Build.VERSION.SDK_INT >= 31) dynamicDarkColorScheme(context) else darkColorScheme()
            MaterialTheme(colorScheme = colors) {
                Surface(modifier = Modifier.fillMaxSize(), color = MaterialTheme.colorScheme.background) {
                    App()
                }
            }
        }
    }
}

@Composable
fun App(vm: MainViewModel = viewModel()) {
    val s by vm.state.collectAsStateWithLifecycle()
    var prompt by rememberSaveable { mutableStateOf("portrait of a lovely cat") }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .safeDrawingPadding()
            .imePadding()
            .verticalScroll(rememberScrollState())
            .padding(horizontal = 20.dp, vertical = 16.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp),
    ) {
        Header(s)

        if (!s.modelsReady) {
            DownloadCard(s, onDownload = vm::download, onCancel = vm::cancelDownload)
        } else {
            ImagePanel(s)
            TimingLine(s)
            OutlinedTextField(
                value = prompt,
                onValueChange = { prompt = it },
                label = { Text("Prompt") },
                modifier = Modifier.fillMaxWidth(),
                minLines = 2,
                maxLines = 4,
            )
            EnginePicker(s, onPick = vm::setEngine)
            Button(
                onClick = { vm.generate(prompt) },
                enabled = s.loadedEngine != null && !s.generating && prompt.isNotBlank(),
                modifier = Modifier.fillMaxWidth().height(52.dp),
            ) {
                Icon(Icons.Outlined.AutoAwesome, contentDescription = null)
                Spacer(Modifier.width(8.dp))
                Text(
                    when {
                        s.loading -> "Loading model…"
                        s.generating -> "Generating…"
                        else -> "Generate"
                    }
                )
            }
            AnimatedVisibility(visible = s.image != null && !s.generating) {
                FilledTonalButton(onClick = vm::save, modifier = Modifier.fillMaxWidth()) {
                    Icon(Icons.Outlined.Download, contentDescription = null)
                    Spacer(Modifier.width(8.dp))
                    Text("Save to gallery")
                }
            }
        }

        s.message?.let { Text(it, color = MaterialTheme.colorScheme.primary, style = MaterialTheme.typography.bodyMedium) }
        s.error?.let { ErrorCard(it) }
    }
}

@Composable
fun Header(s: UiState) {
    Column {
        Text("SD NPU", style = MaterialTheme.typography.headlineMedium, fontWeight = FontWeight.SemiBold)
        Text(
            "Stable Diffusion (SDXS) with the UNet and decoder on the Snapdragon NPU",
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
        Spacer(Modifier.height(10.dp))
        Row(verticalAlignment = Alignment.CenterVertically) {
            Icon(Icons.Outlined.Memory, contentDescription = null, modifier = Modifier.size(18.dp),
                tint = if (s.npuSupported) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.error)
            Spacer(Modifier.width(6.dp))
            Text(
                "${Build.MODEL} · ${s.soc}" + if (s.npuSupported) " · NPU supported" else " · NPU files are for ${Models.NPU_SOC}, CPU only",
                style = MaterialTheme.typography.labelLarge,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
    }
}

@Composable
fun DownloadCard(s: UiState, onDownload: () -> Unit, onCancel: () -> Unit) {
    val context = LocalContext.current
    Card(colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceContainerHigh)) {
        Column(Modifier.padding(20.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Icon(Icons.Outlined.CloudDownload, contentDescription = null, tint = MaterialTheme.colorScheme.primary)
                Spacer(Modifier.width(10.dp))
                Text("Model files needed", style = MaterialTheme.typography.titleMedium)
            }
            Text(
                "SDXS from Hugging Face, plus its UNet and decoder compiled for the ${Models.NPU_SOC} NPU: " +
                    "%.1f GB, downloaded once in the background. Wi-Fi recommended.".format(Models.totalBytes / 1e9),
                style = MaterialTheme.typography.bodyMedium,
            )
            val d = s.download
            if (s.verifying) {
                LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
                Text("Checking the files…", style = MaterialTheme.typography.labelMedium)
            } else if (d != null && (d.running || d.done > 0)) {
                LinearProgressIndicator(
                    progress = { (d.done.toFloat() / d.total).coerceIn(0f, 1f) },
                    modifier = Modifier.fillMaxWidth(),
                )
                Text(
                    "%.0f of %.0f MB".format(d.done / 1e6, d.total / 1e6),
                    style = MaterialTheme.typography.labelMedium,
                    fontFamily = FontFamily.Monospace,
                )
            }
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                if (s.verifying) {
                    // nothing to do while the files are checked
                } else if (d?.running == true) {
                    TextButton(onClick = onCancel) { Text("Cancel") }
                } else {
                    Button(onClick = onDownload) { Text(if ((d?.done ?: 0) > 0) "Resume download" else "Download") }
                }
            }
            Text(
                "Files go to ${Models.dir(context).path}; you can also copy them there over USB (see the app's README).",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
    }
}

@Composable
fun ImagePanel(s: UiState) {
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .aspectRatio(1f)
            .clip(RoundedCornerShape(20.dp))
            .background(MaterialTheme.colorScheme.surfaceContainerHigh),
        contentAlignment = Alignment.Center,
    ) {
        s.image?.let {
            Image(bitmap = it.asImageBitmap(), contentDescription = "Generated image", modifier = Modifier.fillMaxSize())
        }
        if (s.generating || s.loading) {
            Box(Modifier.fillMaxSize().background(Color.Black.copy(alpha = 0.35f)), contentAlignment = Alignment.Center) {
                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    CircularProgressIndicator()
                    Spacer(Modifier.height(12.dp))
                    Text(
                        if (s.loading) "Loading ${s.engine.label} model…" else "Generating on the ${s.loadedEngine?.label}…",
                        color = Color.White,
                    )
                }
            }
        } else if (s.image == null) {
            Text("Your image appears here", color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
    }
}

@Composable
fun TimingLine(s: UiState) {
    val t = s.timings ?: return
    Text(
        "${t.engine.label} · text %.0f ms · UNet %.0f ms · decoder %.0f ms · total %.2f s · seed %d".format(
            t.textMs, t.unetMs, t.decodeMs, t.totalMs / 1000, s.seed,
        ),
        style = MaterialTheme.typography.labelMedium,
        fontFamily = FontFamily.Monospace,
        color = MaterialTheme.colorScheme.onSurfaceVariant,
    )
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun EnginePicker(s: UiState, onPick: (Engine) -> Unit) {
    SingleChoiceSegmentedButtonRow(modifier = Modifier.fillMaxWidth()) {
        Engine.entries.forEachIndexed { i, e ->
            SegmentedButton(
                selected = s.engine == e,
                onClick = { onPick(e) },
                enabled = !s.loading && !s.generating && (e != Engine.NPU || s.npuSupported),
                shape = SegmentedButtonDefaults.itemShape(index = i, count = Engine.entries.size),
            ) {
                Text(if (e == Engine.NPU) "NPU (Hexagon)" else "CPU (ggml)")
            }
        }
    }
}

@Composable
fun ErrorCard(text: String) {
    Card(colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.errorContainer)) {
        Text(
            text,
            modifier = Modifier.padding(16.dp),
            color = MaterialTheme.colorScheme.onErrorContainer,
            style = MaterialTheme.typography.bodyMedium,
        )
    }
}
