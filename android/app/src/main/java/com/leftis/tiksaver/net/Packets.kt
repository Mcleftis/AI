package com.leftis.tiksaver.net

/**
 * Minimal IPv4 / TCP / UDP header handling.
 *
 * Everything here works directly on [ByteArray] with explicit offsets: the tun
 * device hands us one whole IP packet per read, and building a reply is cheaper
 * as a single allocation than as a tree of objects.
 */
internal object Proto {
    const val TCP = 6
    const val UDP = 17

    const val IP_HDR = 20
    const val TCP_HDR = 20
    const val UDP_HDR = 8

    /** Payload we are willing to put in one segment towards the app. */
    const val MSS = 1400

    const val FIN = 0x01
    const val SYN = 0x02
    const val RST = 0x04
    const val PSH = 0x08
    const val ACK = 0x10
    const val URG = 0x20
}

internal fun ByteArray.u8(i: Int): Int = this[i].toInt() and 0xFF

internal fun ByteArray.u16(i: Int): Int = (u8(i) shl 8) or u8(i + 1)

internal fun ByteArray.u32(i: Int): Long =
    (u16(i).toLong() shl 16) or u16(i + 2).toLong()

internal fun ByteArray.putU16(i: Int, v: Int) {
    this[i] = ((v ushr 8) and 0xFF).toByte()
    this[i + 1] = (v and 0xFF).toByte()
}

internal fun ByteArray.putU32(i: Int, v: Long) {
    putU16(i, ((v ushr 16) and 0xFFFFL).toInt())
    putU16(i + 2, (v and 0xFFFFL).toInt())
}

/** IPv4 addresses are carried as raw 32-bit ints in network byte order. */
internal fun ipToString(addr: Int): String =
    "${(addr ushr 24) and 0xFF}.${(addr ushr 16) and 0xFF}.${(addr ushr 8) and 0xFF}.${addr and 0xFF}"

internal fun ipToBytes(addr: Int): ByteArray = byteArrayOf(
    ((addr ushr 24) and 0xFF).toByte(),
    ((addr ushr 16) and 0xFF).toByte(),
    ((addr ushr 8) and 0xFF).toByte(),
    (addr and 0xFF).toByte(),
)

internal fun bytesToIp(b: ByteArray): Int =
    ((b[0].toInt() and 0xFF) shl 24) or
        ((b[1].toInt() and 0xFF) shl 16) or
        ((b[2].toInt() and 0xFF) shl 8) or
        (b[3].toInt() and 0xFF)

/** Identifies one flow as seen from the app's point of view. */
internal data class FlowKey(
    val protocol: Int,
    val srcIp: Int,
    val srcPort: Int,
    val dstIp: Int,
    val dstPort: Int,
) {
    override fun toString(): String =
        "${ipToString(srcIp)}:$srcPort > ${ipToString(dstIp)}:$dstPort/${if (protocol == Proto.TCP) "tcp" else "udp"}"
}

/** Read-only view over an IPv4 packet sitting in [data]. */
internal class Ipv4Packet(val data: ByteArray, val length: Int) {
    val version: Int get() = data.u8(0) ushr 4
    val headerLength: Int get() = (data.u8(0) and 0x0F) * 4
    val totalLength: Int get() = data.u16(2)
    val protocol: Int get() = data.u8(9)
    val srcIp: Int get() = data.u32(12).toInt()
    val dstIp: Int get() = data.u32(16).toInt()

    /** True for a fragment that is not the whole datagram. */
    val isFragment: Boolean
        get() {
            val flagsAndOffset = data.u16(6)
            val moreFragments = (flagsAndOffset and 0x2000) != 0
            val offset = flagsAndOffset and 0x1FFF
            return moreFragments || offset != 0
        }

    val payloadOffset: Int get() = headerLength

    /**
     * Length of the transport payload according to the IP header, clamped to
     * what we actually read, so a lying header can never push us out of bounds.
     */
    val payloadLength: Int
        get() = (minOf(totalLength, length) - headerLength).coerceAtLeast(0)
}

/** Ones' complement checksum over [len] bytes starting at [off], plus [initial]. */
private fun checksum(data: ByteArray, off: Int, len: Int, initial: Long): Int {
    var sum = initial
    var i = off
    val end = off + len - 1
    while (i < end) {
        sum += data.u16(i).toLong()
        i += 2
    }
    if (i == end) sum += (data.u8(i) shl 8).toLong()
    while ((sum ushr 16) != 0L) sum = (sum and 0xFFFF) + (sum ushr 16)
    return (sum.inv() and 0xFFFF).toInt()
}

private fun pseudoHeaderSum(srcIp: Int, dstIp: Int, protocol: Int, transportLen: Int): Long {
    var sum = 0L
    sum += ((srcIp ushr 16) and 0xFFFF).toLong()
    sum += (srcIp and 0xFFFF).toLong()
    sum += ((dstIp ushr 16) and 0xFFFF).toLong()
    sum += (dstIp and 0xFFFF).toLong()
    sum += protocol.toLong()
    sum += transportLen.toLong()
    return sum
}

internal object PacketBuilder {

    private var ipId = 1

    private fun nextId(): Int {
        ipId = (ipId + 1) and 0xFFFF
        return ipId
    }

    private fun writeIpv4Header(
        out: ByteArray,
        srcIp: Int,
        dstIp: Int,
        protocol: Int,
        totalLength: Int,
    ) {
        out[0] = 0x45
        out[1] = 0
        out.putU16(2, totalLength)
        out.putU16(4, nextId())
        out.putU16(6, 0x4000) // don't fragment
        out[8] = 64 // ttl
        out[9] = protocol.toByte()
        out.putU16(10, 0)
        out.putU32(12, srcIp.toLong() and 0xFFFFFFFFL)
        out.putU32(16, dstIp.toLong() and 0xFFFFFFFFL)
        out.putU16(10, checksum(out, 0, Proto.IP_HDR, 0L))
    }

    /**
     * Build a TCP segment travelling towards the app, so [srcIp]/[srcPort] are
     * the remote peer and [dstIp]/[dstPort] are the app.
     */
    fun tcp(
        srcIp: Int,
        srcPort: Int,
        dstIp: Int,
        dstPort: Int,
        seq: Long,
        ack: Long,
        flags: Int,
        window: Int,
        payload: ByteArray? = null,
        payloadOffset: Int = 0,
        payloadLength: Int = 0,
        mssOption: Int = 0,
    ): ByteArray {
        val optionLen = if (mssOption > 0) 4 else 0
        val tcpLen = Proto.TCP_HDR + optionLen + payloadLength
        val total = Proto.IP_HDR + tcpLen
        val out = ByteArray(total)

        writeIpv4Header(out, srcIp, dstIp, Proto.TCP, total)

        val t = Proto.IP_HDR
        out.putU16(t, srcPort)
        out.putU16(t + 2, dstPort)
        out.putU32(t + 4, seq and 0xFFFFFFFFL)
        out.putU32(t + 8, ack and 0xFFFFFFFFL)
        out[t + 12] = (((Proto.TCP_HDR + optionLen) / 4) shl 4).toByte()
        out[t + 13] = flags.toByte()
        out.putU16(t + 14, window.coerceIn(0, 0xFFFF))
        out.putU16(t + 16, 0)
        out.putU16(t + 18, 0)

        if (optionLen > 0) {
            out[t + 20] = 2 // kind: MSS
            out[t + 21] = 4 // length
            out.putU16(t + 22, mssOption)
        }

        if (payload != null && payloadLength > 0) {
            System.arraycopy(payload, payloadOffset, out, t + Proto.TCP_HDR + optionLen, payloadLength)
        }

        val sum = checksum(out, t, tcpLen, pseudoHeaderSum(srcIp, dstIp, Proto.TCP, tcpLen))
        // A zero checksum is legal for TCP, no 0xFFFF fixup needed (that rule is UDP-only).
        out.putU16(t + 16, sum)
        return out
    }

    /** Build a UDP datagram travelling towards the app. */
    fun udp(
        srcIp: Int,
        srcPort: Int,
        dstIp: Int,
        dstPort: Int,
        payload: ByteArray,
        payloadOffset: Int = 0,
        payloadLength: Int = payload.size,
    ): ByteArray {
        val udpLen = Proto.UDP_HDR + payloadLength
        val total = Proto.IP_HDR + udpLen
        val out = ByteArray(total)

        writeIpv4Header(out, srcIp, dstIp, Proto.UDP, total)

        val u = Proto.IP_HDR
        out.putU16(u, srcPort)
        out.putU16(u + 2, dstPort)
        out.putU16(u + 4, udpLen)
        out.putU16(u + 6, 0)
        System.arraycopy(payload, payloadOffset, out, u + Proto.UDP_HDR, payloadLength)

        var sum = checksum(out, u, udpLen, pseudoHeaderSum(srcIp, dstIp, Proto.UDP, udpLen))
        // In UDP, 0 means "no checksum", so it has to be sent as all ones instead.
        if (sum == 0) sum = 0xFFFF
        out.putU16(u + 6, sum)
        return out
    }

    /**
     * ICMP "destination unreachable / port unreachable" for [original].
     *
     * Sent when we refuse a UDP datagram (QUIC) so the app fails over to TCP
     * immediately instead of sitting through its own probe timeout.
     */
    fun icmpPortUnreachable(original: Ipv4Packet): ByteArray? {
        val srcIp = original.dstIp
        val dstIp = original.srcIp
        // RFC 792: the ICMP payload is the offending IP header plus 8 bytes.
        val quoted = minOf(original.headerLength + 8, original.length)
        if (quoted < Proto.IP_HDR) return null

        val icmpLen = 8 + quoted
        val total = Proto.IP_HDR + icmpLen
        val out = ByteArray(total)
        writeIpv4Header(out, srcIp, dstIp, PROTO_ICMP, total)

        val c = Proto.IP_HDR
        out[c] = 3 // destination unreachable
        out[c + 1] = 3 // port unreachable
        out.putU16(c + 2, 0)
        out.putU32(c + 4, 0)
        System.arraycopy(original.data, 0, out, c + 8, quoted)
        // ICMP has no pseudo header.
        out.putU16(c + 2, checksum(out, c, icmpLen, 0L))
        return out
    }

    const val PROTO_ICMP = 1
}

/** Signed 32-bit-wrapping difference, i.e. `a - b` in TCP sequence space. */
internal fun seqDiff(a: Long, b: Long): Long {
    var d = (a - b) and 0xFFFFFFFFL
    if (d >= 0x80000000L) d -= 0x100000000L
    return d
}

internal fun seqAdd(a: Long, n: Long): Long = (a + n) and 0xFFFFFFFFL
