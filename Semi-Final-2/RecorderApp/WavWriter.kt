package com.example.vocalizerec

class WavWriter {

    fun makeWavHeader(pcmDataLength: Long, sampleRate: Int): ByteArray {

        val channels = 1
        val byteRate = sampleRate * 2 * channels
        val totalDataLen = pcmDataLength + 36

        val header = ByteArray(44)

        header[0] = 'R'.code.toByte()
        header[1] = 'I'.code.toByte()
        header[2] = 'F'.code.toByte()
        header[3] = 'F'.code.toByte()

        header[4] = (totalDataLen and 0xff).toByte()
        header[5] = ((totalDataLen shr 8) and 0xff).toByte()
        header[6] = ((totalDataLen shr 16) and 0xff).toByte()
        header[7] = ((totalDataLen shr 24) and 0xff).toByte()

        header[8] = 'W'.code.toByte()
        header[9] = 'A'.code.toByte()
        header[10] = 'V'.code.toByte()
        header[11] = 'E'.code.toByte()

        header[12] = 'f'.code.toByte()
        header[13] = 'm'.code.toByte()
        header[14] = 't'.code.toByte()
        header[15] = ' '.code.toByte()

        header[16] = 16
        header[17] = 0
        header[18] = 0
        header[19] = 0
        header[20] = 1
        header[21] = 0
        header[22] = 1
        header[23] = 0

        header[24] = (sampleRate).toByte()
        header[25] = ((sampleRate shr 8)).toByte()
        header[26] = ((sampleRate shr 16)).toByte()
        header[27] = ((sampleRate shr 24)).toByte()

        header[28] = (byteRate).toByte()
        header[29] = ((byteRate shr 8)).toByte()
        header[30] = ((byteRate shr 16)).toByte()
        header[31] = ((byteRate shr 24)).toByte()

        header[32] = 2
        header[33] = 0
        header[34] = 16
        header[35] = 0

        header[36] = 'd'.code.toByte()
        header[37] = 'a'.code.toByte()
        header[38] = 't'.code.toByte()
        header[39] = 'a'.code.toByte()

        header[40] = (pcmDataLength).toByte()
        header[41] = ((pcmDataLength shr 8)).toByte()
        header[42] = ((pcmDataLength shr 16)).toByte()
        header[43] = ((pcmDataLength shr 24)).toByte()

        return header
    }
}
