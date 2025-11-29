package com.example.vocalizerec

import android.Manifest
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Bundle
import android.widget.Button
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import java.io.File
import java.io.FileOutputStream

class MainActivity : AppCompatActivity() {

    private lateinit var tvStatus: TextView
    private lateinit var btnRecord: Button

    private var selectedWord = "Base"

    private val sampleRate = 16000
    private val bufferSize = AudioRecord.getMinBufferSize(
        sampleRate,
        AudioFormat.CHANNEL_IN_MONO,
        AudioFormat.ENCODING_PCM_16BIT
    )

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        tvStatus = findViewById(R.id.tvStatus)
        btnRecord = findViewById(R.id.btnRecord)

        requestPermissions()

        findViewById<Button>(R.id.btnBase).setOnClickListener { selectedWord = "Base" }
        findViewById<Button>(R.id.btnOK).setOnClickListener { selectedWord = "OK" }
        findViewById<Button>(R.id.btnHato).setOnClickListener { selectedWord = "Hato" }
        findViewById<Button>(R.id.btnKhana).setOnClickListener { selectedWord = "Khana" }
        findViewById<Button>(R.id.btnJao).setOnClickListener { selectedWord = "Jao" }
        findViewById<Button>(R.id.btnNahi).setOnClickListener { selectedWord = "Nahi" }
        findViewById<Button>(R.id.btnHaan).setOnClickListener { selectedWord = "Haan" }

        btnRecord.setOnClickListener {
            tvStatus.text = "Recording $selectedWord ..."
            Thread { recordWord() }.start()
        }
    }

    private fun requestPermissions() {
        val needed = arrayOf(Manifest.permission.RECORD_AUDIO)

        val missing = needed.any {
            ContextCompat.checkSelfPermission(this, it) != PackageManager.PERMISSION_GRANTED
        }

        if (missing) {
            ActivityCompat.requestPermissions(this, needed, 101)
        }
    }

    private fun recordWord() {

        val recorder = AudioRecord(
            MediaRecorder.AudioSource.VOICE_COMMUNICATION,
            sampleRate,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
            bufferSize
        )

        val pcmBytes = ByteArray(sampleRate * 2 * 2) // 2 sec
        recorder.startRecording()
        recorder.read(pcmBytes, 0, pcmBytes.size)
        recorder.stop()
        recorder.release()

        saveToFile(pcmBytes)
    }

    private fun saveToFile(pcmData: ByteArray) {

        val folder = File(getExternalFilesDir(null), selectedWord)
        folder.mkdirs()

        val index = folder.listFiles()?.size ?: 0
        val file = File(folder, "${selectedWord}_${index}.wav")

        val header = WavWriter().makeWavHeader(pcmData.size.toLong(), sampleRate)

        val out = FileOutputStream(file)
        out.write(header)
        out.write(pcmData)
        out.close()

        runOnUiThread {
            tvStatus.text = "Saved: ${file.absolutePath}"
        }
    }
}
