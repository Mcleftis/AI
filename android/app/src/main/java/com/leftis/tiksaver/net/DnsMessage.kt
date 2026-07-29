package com.leftis.tiksaver.net

/**
 * Just enough DNS to read the question and synthesise an NXDOMAIN answer.
 * Everything that is not blocked is relayed verbatim to the upstream resolver,
 * so we never have to build real answers.
 */
internal object DnsMessage {

    private const val HEADER = 12

    /** Returns the queried name, or null if this is not a single-question query. */
    fun questionName(buf: ByteArray, length: Int): String? {
        if (length < HEADER + 5) return null
        val flags = buf.u16(2)
        if (flags and 0x8000 != 0) return null // already a response
        if (buf.u16(4) != 1) return null // exactly one question, which is all TikTok sends

        val sb = StringBuilder()
        var p = HEADER
        while (p < length) {
            val len = buf.u8(p)
            if (len == 0) return sb.toString().ifEmpty { null }
            // Compression pointers are not legal in a question; bail out rather
            // than chase them.
            if (len and 0xC0 != 0) return null
            if (p + 1 + len > length) return null
            if (sb.isNotEmpty()) sb.append('.')
            sb.append(String(buf, p + 1, len, Charsets.US_ASCII))
            p += 1 + len
            if (sb.length > 255) return null
        }
        return null
    }

    /** End offset of the question section, or -1 when it is malformed. */
    private fun questionEnd(buf: ByteArray, length: Int): Int {
        var p = HEADER
        while (p < length) {
            val len = buf.u8(p)
            if (len == 0) {
                val end = p + 1 + 4 // null label + qtype + qclass
                return if (end <= length) end else -1
            }
            if (len and 0xC0 != 0) return -1
            p += 1 + len
        }
        return -1
    }

    /** Builds an NXDOMAIN reply that mirrors the question of [query]. */
    fun nxdomain(query: ByteArray, length: Int): ByteArray? {
        val end = questionEnd(query, length)
        if (end < 0) return null

        val out = ByteArray(end)
        System.arraycopy(query, 0, out, 0, end)

        val queryFlags = query.u16(2)
        val responseFlags = 0x8000 or          // QR = response
            (queryFlags and 0x7800) or         // preserve opcode
            (queryFlags and 0x0100) or         // preserve RD
            0x0080 or                          // RA = recursion available
            0x0003                             // RCODE = NXDOMAIN
        out.putU16(2, responseFlags)
        out.putU16(6, 0) // ANCOUNT
        out.putU16(8, 0) // NSCOUNT
        out.putU16(10, 0) // ARCOUNT
        return out
    }
}
