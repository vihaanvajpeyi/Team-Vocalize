package com.example.vocalize

import android.Manifest
import android.content.ComponentName
import android.content.Intent
import android.content.ServiceConnection
import android.content.pm.PackageManager
import android.media.AudioManager
import android.os.Bundle
import android.os.IBinder
import android.speech.tts.TextToSpeech
import android.util.Log
import android.widget.Button
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import java.util.Locale
import java.util.ArrayDeque
import kotlin.math.sqrt

class MainActivity : AppCompatActivity() {
    private val TAG = "MainActivity"

    private lateinit var listenBtn: Button
    private lateinit var predictionText: TextView
    private lateinit var logText: TextView
    private lateinit var waveformView: WaveformView

    private var audioCapture: AudioCapture? = null
    private var classifier: TFLiteClassifier? = null

    private lateinit var tts: TextToSpeech
    private var listening = false

    private val labels: List<String> by lazy {
        assets.open("labels.txt").bufferedReader().useLines { it.toList() }
    }

    // smoothing / hysteresis parameters
    private val SMOOTHING_WINDOW = 5
    private val CONFIDENCE_THRESHOLD = 0.60f
    private val VAD_THRESHOLD_INIT = 0.005f // initial RMS threshold (calibrate)
    private var vadThreshold = VAD_THRESHOLD_INIT
    private val labelWindow = ArrayDeque<String>()
    private val probWindow = ArrayDeque<FloatArray>()

    // foreground service connection
    private var serviceBound = false
    private var svcConnection: ServiceConnection? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        listenBtn = findViewById(R.id.listenBtn)
        predictionText = findViewById(R.id.predictionText)
        logText = findViewById(R.id.logText)
        waveformView = findViewById(R.id.waveformView)

        CoroutineScope(Dispatchers.Default).launch {
            try {
                classifier = TFLiteClassifier(this@MainActivity, "model.tflite", labels)
                val n = classifier?.getNumClasses() ?: -1
                log("Model loaded, classes=$n")
            } catch (e: Exception) {
                log("Model load failed: ${e.message}")
            }
        }

        tts = TextToSpeech(this) { status ->
            if (status == TextToSpeech.SUCCESS) log("TTS ready")
            else log("TTS init failed")
        }

        listenBtn.setOnClickListener { toggleListening() }
    }

    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) startListening() else log("Record permission denied")
    }

    private fun toggleListening() {
        if (!listening) {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
                requestPermissionLauncher.launch(Manifest.permission.RECORD_AUDIO)
                return
            }
            startListening()
        } else {
            stopListening()
        }
    }

    private fun startListening() {
        if (classifier == null) { log("Classifier not loaded"); return }
        listening = true
        listenBtn.text = "Listening"

        // Start foreground service to keep process alive
        val svcIntent = Intent(this, RecordingForegroundService::class.java)
        ContextCompat.startForegroundService(this, svcIntent)
        // Optionally bind to service if needed
        svcConnection = object: ServiceConnection {
            override fun onServiceConnected(name: ComponentName?, binder: IBinder?) {
                serviceBound = true
            }
            override fun onServiceDisconnected(name: ComponentName?) { serviceBound = false }
        }
        // bindService(svcIntent, svcConnection!!, BIND_AUTO_CREATE)

        // AudioCapture emits 200ms windows; we will run inference every emit and maintain smoothing
        audioCapture = AudioCapture({ window ->
            waveformView.updateSamples(window)
            // compute RMS for VAD gating
            val rms = computeRMS(window)
            // adaptive calibration: when not listening or low energy, slowly adapt vadThreshold
            adaptVadThreshold(rms)
            if (rms < vadThreshold) {
                // treat as background/no-speech
                pushResult("BGN", FloatArray(classifier!!.getNumClasses()) { 0f })
                return@AudioCapture
            }
            // otherwise run inference
            CoroutineScope(Dispatchers.Default).launch {
                try {
                    val (label, probs) = classifier!!.predict(window)
                    pushResult(label, probs)
                } catch (e: Exception) {
                    log("Inference error: ${e.message}")
                }
            }
        }, sampleRate = 16000, emitMs = 200L)

        // route audio to bluetooth SCO if available (best-effort)
        routeAudioToBluetoothIfAvailable()

        audioCapture?.start()
        log("Started listening (foreground service active)")
    }

    private fun stopListening() {
        listening = false
        listenBtn.text = "Start Listening"
        audioCapture?.stop()
        stopBluetoothScoIfNeeded()
        // stop foreground service
        val svcIntent = Intent(this, RecordingForegroundService::class.java)
        stopService(svcIntent)
        // unbind if bound
        try {
            if (svcConnection != null) unbindService(svcConnection!!)
        } catch (_: Exception) {}
        svcConnection = null
        serviceBound = false
        log("Stopped listening")
    }

    private fun pushResult(label: String, probs: FloatArray) {
        // maintain sliding windows
        synchronized(this) {
            labelWindow.addLast(label)
            probWindow.addLast(probs)
            if (labelWindow.size > SMOOTHING_WINDOW) { labelWindow.removeFirst(); probWindow.removeFirst() }
        }
        // smooth: majority vote and average probs
        val smoothed = computeSmoothedLabel()
        runOnUiThread {
            if (smoothed.first == "BGN") predictionText.text = "-" else predictionText.text = smoothed.first
        }
        // TTS: speak only when above confidence and when label changed (debounce)
        if (smoothed.second >= CONFIDENCE_THRESHOLD && smoothed.first != "BGN") {
            speakLabelDebounced(smoothed.first)
        }
    }

    // compute RMS
    private fun computeRMS(window: FloatArray): Float {
        var s = 0.0
        for (v in window) s += (v * v).toDouble()
        val rms = sqrt(s / window.size).toFloat()
        return rms
    }

    // adaptively update vadThreshold toward quiet baseline when low energy
    private var quietSamples = 0
    private fun adaptVadThreshold(rms: Float) {
        // if we detect long silence, slowly lower vad threshold; if loud, increase quickly
        if (rms < vadThreshold) {
            quietSamples = (quietSamples + 1).coerceAtMost(50)
            val target = (vadThreshold * 0.99f + rms * 0.01f)
            vadThreshold = vadThreshold * 0.995f + target * 0.005f
        } else {
            // speech present - raise threshold to avoid noise triggers
            vadThreshold = vadThreshold * 0.95f + rms * 0.05f
            quietSamples = 0
        }
    }

    // compute smoothed label and max confidence (average probs)
    private fun computeSmoothedLabel(): Pair<String, Float> {
        val counts = HashMap<String, Int>()
        val avgProbs = FloatArray(classifier!!.getNumClasses())
        synchronized(this) {
            for (p in probWindow) {
                for (i in avgProbs.indices) avgProbs[i] += p[i] / probWindow.size
            }
            for (l in labelWindow) counts[l] = counts.getOrDefault(l, 0) + 1
        }
        // majority label
        var bestLabel = "BGN"; var bestCount = -1
        for ((k, v) in counts) if (v > bestCount) { bestLabel = k; bestCount = v }
        // find avg prob for bestLabel
        val idx = labels.indexOf(bestLabel).coerceAtLeast(0)
        val conf = if (avgProbs.isNotEmpty() && idx < avgProbs.size) avgProbs[idx] else 0f
        return Pair(bestLabel, conf)
    }

    // simple TTS debounce to avoid rapid repeats
    private var lastSpokenLabel: String? = null
    private var lastSpokenTime = 0L
    private fun speakLabelDebounced(label: String) {
        val now = System.currentTimeMillis()
        if (label == lastSpokenLabel && now - lastSpokenTime < 1200) return // don't repeat within 1.2s
        lastSpokenLabel = label; lastSpokenTime = now
        speakLabel(label)
    }

    private fun speakLabel(label: String) {
        if (label == "BGN") return
        val utter = label
        val locale = if (label == "Ok") Locale("en", "IN") else Locale("hi", "IN")
        val res = tts.setLanguage(locale)
        if (res == TextToSpeech.LANG_MISSING_DATA || res == TextToSpeech.LANG_NOT_SUPPORTED) log("TTS locale not supported: $locale")
        tts.setSpeechRate(1.0f); tts.setPitch(1.0f)
        tts.speak(utter, TextToSpeech.QUEUE_ADD, null, "vocalize_tts")
    }

    private fun routeAudioToBluetoothIfAvailable() {
        val audioManager = getSystemService(AUDIO_SERVICE) as AudioManager
        try {
            if (!audioManager.isBluetoothScoOn) audioManager.startBluetoothSco()
            audioManager.mode = AudioManager.MODE_IN_COMMUNICATION
            log("Requested Bluetooth SCO")
        } catch (e: Exception) {
            log("Audio routing failed: ${e.message}")
        }
    }

    private fun stopBluetoothScoIfNeeded() {
        val audioManager = getSystemService(AUDIO_SERVICE) as AudioManager
        try {
            if (audioManager.isBluetoothScoOn) audioManager.stopBluetoothSco()
            audioManager.mode = AudioManager.MODE_NORMAL
        } catch (_: Exception) {}
    }

    private fun log(msg: String) {
        runOnUiThread { logText.append("\n$msg") }
        Log.d(TAG, msg)
    }

    override fun onDestroy() {
        super.onDestroy()
        audioCapture?.stop()
        classifier?.close()
        tts.stop(); tts.shutdown()
    }
}
