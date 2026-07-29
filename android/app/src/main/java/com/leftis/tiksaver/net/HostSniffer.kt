package com.leftis.tiksaver.net

/**
 * Recovers the destination hostname from the first bytes an app writes on a
 * connection: the SNI of a TLS ClientHello, or the `Host:` header of a plain
 * HTTP request.
 *
 * This is what makes hostname filtering survive TikTok's habit of resolving
 * names through its own HTTPDNS service instead of the system resolver — by the
 * time we see the ClientHello, DNS is already out of the picture.
 */
internal object HostSniffer {

    sealed interface Result {
        /** Not enough bytes yet; call again once more data has arrived. */
        data object NeedMore : Result

        /** Definitely not something we can read a hostname out of. */
        data object Unknown : Result

        data class Host(val name: String) : Result
    }

    fun sniff(buf: ByteArray, length: Int): Result {
        if (length < 1) return Result.NeedMore
        return if (buf[0].toInt() == 0x16) parseTls(buf, length) else parseHttp(buf, length)
    }

    // -- TLS -----------------------------------------------------------------

    private fun parseTls(buf: ByteArray, length: Int): Result {
        // TLS record header: type(1) version(2) length(2)
        if (length < 5) return Result.NeedMore
        val recordLen = buf.u16(3)
        val recordEnd = 5 + recordLen
        if (recordEnd > length) return Result.NeedMore
        if (recordLen < 4) return Result.Unknown

        var p = 5
        if (buf.u8(p) != 0x01) return Result.Unknown // not a ClientHello
        val handshakeLen = (buf.u8(p + 1) shl 16) or buf.u16(p + 2)
        p += 4
        val handshakeEnd = minOf(p + handshakeLen, recordEnd)

        p += 2 // legacy_version
        p += 32 // random
        if (p >= handshakeEnd) return Result.Unknown

        val sessionIdLen = buf.u8(p); p += 1 + sessionIdLen
        if (p + 2 > handshakeEnd) return Result.Unknown

        val cipherSuitesLen = buf.u16(p); p += 2 + cipherSuitesLen
        if (p + 1 > handshakeEnd) return Result.Unknown

        val compressionLen = buf.u8(p); p += 1 + compressionLen
        if (p + 2 > handshakeEnd) return Result.Unknown

        val extensionsLen = buf.u16(p); p += 2
        val extensionsEnd = minOf(p + extensionsLen, handshakeEnd)

        while (p + 4 <= extensionsEnd) {
            val type = buf.u16(p)
            val len = buf.u16(p + 2)
            val body = p + 4
            if (body + len > extensionsEnd) return Result.Unknown
            if (type == 0x0000) return parseServerNameExtension(buf, body, len)
            p = body + len
        }
        return Result.Unknown
    }

    private fun parseServerNameExtension(buf: ByteArray, off: Int, len: Int): Result {
        if (len < 5) return Result.Unknown
        // server_name_list length(2), then entries of type(1) length(2) name
        var p = off + 2
        val end = off + len
        while (p + 3 <= end) {
            val nameType = buf.u8(p)
            val nameLen = buf.u16(p + 1)
            val nameStart = p + 3
            if (nameStart + nameLen > end) return Result.Unknown
            if (nameType == 0x00) {
                return Result.Host(String(buf, nameStart, nameLen, Charsets.US_ASCII))
            }
            p = nameStart + nameLen
        }
        return Result.Unknown
    }

    // -- HTTP ----------------------------------------------------------------

    private val HTTP_METHODS = listOf("GET ", "POST", "HEAD", "PUT ", "OPTI", "DELE", "PATC", "CONN", "TRAC")

    private fun parseHttp(buf: ByteArray, length: Int): Result {
        if (length < 4) return Result.NeedMore
        val head = String(buf, 0, 4, Charsets.US_ASCII)
        if (HTTP_METHODS.none { it.equals(head, ignoreCase = true) }) return Result.Unknown

        val text = String(buf, 0, minOf(length, MAX_HTTP_SCAN), Charsets.ISO_8859_1)
        val headerEnd = text.indexOf("\r\n\r\n")
        val searchIn = if (headerEnd >= 0) text.substring(0, headerEnd) else text

        for (line in searchIn.split("\r\n")) {
            if (line.length > 5 && line.regionMatches(0, "Host:", 0, 5, ignoreCase = true)) {
                val value = line.substring(5).trim().substringBefore(':')
                if (value.isNotEmpty()) return Result.Host(value)
            }
        }
        return if (headerEnd >= 0 || length >= MAX_HTTP_SCAN) Result.Unknown else Result.NeedMore
    }

    private const val MAX_HTTP_SCAN = 4096
}
