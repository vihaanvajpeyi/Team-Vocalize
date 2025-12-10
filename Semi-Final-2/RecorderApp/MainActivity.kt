package com.example.vocalizerec

import android.Manifest
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Bundle
import android.widget.Button
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import java.io.File
import java.io.FileOutputStream

class MainActivity : AppCompatActivity() {

    private lateinit var waveform: WaveformView
    private var selectedWord: String? = null
    private val sampleRate = 16000
    private val wordButtons = mutableListOf<Button>()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        waveform = findViewById(R.id.waveformView)

        requestMicPermission()
        setupWordButtons()
        setupRecordButton()
    }

    private fun requestMicPermission() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
            != PackageManager.PERMISSION_GRANTED
        ) {
            ActivityCompat.requestPermissions(
                this,
                arrayOf(Manifest.permission.RECORD_AUDIO),
                1
            )
        }
    }

    private fun setupWordButtons() {
        val map = mapOf(
            R.id.btnBase to "base",
            R.id.btnOK to "ok",
            R.id.btnHato to "hato",
            R.id.btnKhana to "khana",
            R.id.btnJao to "jao",
            R.id.btnNahi to "nahi",
            R.id.btnHaan to "haan",
            R.id.btnSaat to "saat",
            R.id.btnSaatvik to "saatvik",
            R.id.btnPani to "pani"
        )

        map.forEach { (id, word) ->
            val btn = findViewById<Button>(id)
            wordButtons.add(btn)
            btn.setOnClickListener {
                selectedWord = word
                highlight(btn)
            }
        }
    }

    private fun highlight(active: Button) {
        wordButtons.forEach { it.setBackgroundColor(0xFF444444.toInt()) }
        active.setBackgroundColor(0xFF00AA00.toInt())
    }

    private fun setupRecordButton() {
        val btnRecord = findViewById<Button>(R.id.btnRecord)

        btnRecord.setOnClickListener {
            val word = selectedWord ?: return@setOnClickListener
            Thread {
                val pcm = recordOneSecond()
                saveWav(word, pcm)
            }.start()
        }
    }

    private fun recordOneSecond(): ByteArray {
        val bufferSize = AudioRecord.getMinBufferSize(
            sampleRate,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT
        )

        val recorder = AudioRecord(
            MediaRecorder.AudioSource.MIC,
            sampleRate,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
            bufferSize
        )

        val shortBuffer = ShortArray(1024)
        val totalSamples = sampleRate // 1 second
        val pcmOut = ByteArray(totalSamples * 2)

        recorder.startRecording()

        var written = 0
        while (written < totalSamples) {
            val read = recorder.read(shortBuffer, 0, shortBuffer.size)
            for (i in 0 until read) {
                if (written >= totalSamples) break
                val s = shortBuffer[i]

                waveform.update(s)

                pcmOut[written * 2] = (s.toInt() and 0xFF).toByte()
                pcmOut[written * 2 + 1] = ((s.toInt() shr 8) and 0xFF).toByte()

                written++
            }
        }

        recorder.stop()
        recorder.release()
        return pcmOut
    }

    private fun saveWav(word: String, pcm: ByteArray) {
        val baseDir = File(getExternalFilesDir(null), "dataset/$word")
        if (!baseDir.exists()) baseDir.mkdirs()

        val index = baseDir.listFiles()?.size ?: 0
        val file = File(baseDir, "${word}_${index}.wav")

        val header = WavWriter().makeWavHeader(pcm.size.toLong(), sampleRate)

        FileOutputStream(file).use {
            it.write(header)
            it.write(pcm)
        }
    }
}
