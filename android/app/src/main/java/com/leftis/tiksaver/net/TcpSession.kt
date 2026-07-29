package com.leftis.tiksaver.net

import android.os.SystemClock
import android.util.Log
import java.io.IOException
import java.net.InetAddress
import java.net.InetSocketAddress
import java.nio.ByteBuffer
import java.nio.channels.SelectionKey
import java.nio.channels.SocketChannel
import kotlin.math.min

/**
 * One TCP connection, terminated locally.
 *
 * The app talks TCP to us over the tun device; we talk TCP to the real server
 * over a protected socket and copy payload between the two. Terminating rather
 * than forwarding is what gives us a throttle point: the rate at which we drain
 * the remote socket becomes the rate the app observes, and TCP's own flow
 * control pushes the slowdown back to the server without a single dropped
 * packet.
 *
 * The link towards the app is a file descriptor, not a network, so it does not
 * reorder or lose packets. That lets this implementation stay small: no SACK,
 * no window scaling (deliberately never offered, which disables it in both
 * directions), and a single-segment retransmit timer purely as a safety net.
 */
internal class TcpSession(
    val key: FlowKey,
    private val engine: NatEngine,
) {

    private enum class State { CONNECTING, ESTABLISHED, CLOSED }

    private class Segment(
        val seq: Long,
        val flags: Int,
        val payload: ByteArray?,
        /** Length in sequence space: payload size, or 1 for a bare SYN/FIN. */
        val seqLength: Int,
        var sentAt: Long,
        var tries: Int,
    )

    private var state = State.CONNECTING

    private var channel: SocketChannel? = null
    private var selectionKey: SelectionKey? = null

    /** Next sequence number we will hand out for data flowing to the app. */
    private var sendSeq: Long = (Math.random() * 0xFFFFFFFFL).toLong() and 0xFFFFFFFFL
    private var sendUnack: Long = sendSeq

    /** Next sequence number we expect from the app. */
    private var recvNext: Long = 0

    private var appWindow: Int = 65535

    private val pendingToRemote = ArrayDeque<ByteBuffer>()
    private var pendingToRemoteBytes = 0

    private val unacked = ArrayDeque<Segment>()

    private var appFinReceived = false
    private var finSentToApp = false
    private var shutdownAfterFlush = false

    /**
     * Set when the upload bucket, not the socket, is what stopped the flush.
     * Without this we would arm OP_WRITE on a socket that is perfectly writable
     * and spin the selector until tokens appear.
     */
    private var waitingForUploadTokens = false

    private var sniffBuffer: ByteArray? = null
    private var sniffLength = 0
    private var hostResolved = false

    private var lastActivity = SystemClock.elapsedRealtime()
    private var closingSince = 0L

    val isClosed: Boolean get() = state == State.CLOSED

    // -- lifecycle -----------------------------------------------------------

    /** Handles the initial SYN and kicks off the outbound connection. */
    fun open(clientSeq: Long, window: Int): Boolean {
        recvNext = seqAdd(clientSeq, 1)
        appWindow = window
        return try {
            val ch = SocketChannel.open()
            ch.configureBlocking(false)
            if (!engine.protector.protect(ch.socket())) {
                Log.w(TAG, "protect() refused for $key")
                ch.close()
                return false
            }
            ch.socket().tcpNoDelay = true
            channel = ch
            val target = InetSocketAddress(InetAddress.getByAddress(ipToBytes(key.dstIp)), key.dstPort)
            val connectedImmediately = ch.connect(target)
            selectionKey = ch.register(
                engine.selector,
                if (connectedImmediately) SelectionKey.OP_READ else SelectionKey.OP_CONNECT,
                this,
            )
            if (connectedImmediately) onConnected()
            true
        } catch (e: Exception) {
            Log.w(TAG, "connect failed for $key: ${e.message}")
            closeHard()
            false
        }
    }

    private fun onConnected() {
        state = State.ESTABLISHED
        sendSynAck()
        updateInterest()
    }

    fun onConnectable() {
        val ch = channel ?: return closeHard()
        try {
            if (ch.finishConnect()) onConnected()
        } catch (e: IOException) {
            Log.d(TAG, "connect refused for $key: ${e.message}")
            sendReset()
            closeHard()
        }
    }

    // -- app -> us -----------------------------------------------------------

    /**
     * @param packet the whole IP datagram
     * @param transportOffset offset of the TCP header inside [packet]
     * @param transportLength TCP header plus payload length
     */
    fun onPacket(packet: ByteArray, transportOffset: Int, transportLength: Int) {
        if (state == State.CLOSED) return
        if (transportLength < Proto.TCP_HDR) return

        val t = transportOffset
        val seq = packet.u32(t + 4)
        val ackNum = packet.u32(t + 8)
        val dataOffset = (packet.u8(t + 12) ushr 4) * 4
        val flags = packet.u8(t + 13)
        val window = packet.u16(t + 14)

        if (dataOffset < Proto.TCP_HDR || dataOffset > transportLength) return
        val dataStart = t + dataOffset
        val dataLength = transportLength - dataOffset

        lastActivity = SystemClock.elapsedRealtime()
        appWindow = window

        if (flags and Proto.RST != 0) {
            closeHard()
            return
        }

        // A repeated SYN means our SYN-ACK went missing (or the app restarted
        // the handshake); re-send rather than tearing the session down.
        if (flags and Proto.SYN != 0 && flags and Proto.ACK == 0) {
            if (state == State.ESTABLISHED) retransmitOldest(force = true)
            return
        }

        if (flags and Proto.ACK != 0) onAck(ackNum)
        if (state != State.ESTABLISHED) return

        val segmentEnd = seqAdd(seq, dataLength.toLong())
        var needAck = false

        if (dataLength > 0) {
            when {
                seqDiff(seq, recvNext) == 0L -> {
                    acceptFromApp(packet, dataStart, dataLength)
                    recvNext = segmentEnd
                    needAck = true
                }
                // Already delivered, or a gap we cannot fill without a reassembly
                // queue: re-advertise where we are and let the app retransmit.
                else -> {
                    sendAck()
                    return
                }
            }
        }

        if (flags and Proto.FIN != 0) {
            if (seqDiff(segmentEnd, recvNext) == 0L) {
                recvNext = seqAdd(recvNext, 1)
                onAppFin()
                needAck = true
            } else {
                sendAck()
                return
            }
        }

        if (needAck) sendAck()
        updateInterest()
    }

    private fun acceptFromApp(packet: ByteArray, offset: Int, length: Int) {
        Stats.addUp(length)

        if (engine.config.blockAds && !hostResolved) {
            if (bufferForHostCheck(packet, offset, length)) return
        }

        val copy = ByteArray(length)
        System.arraycopy(packet, offset, copy, 0, length)
        enqueueToRemote(ByteBuffer.wrap(copy))
    }

    /**
     * Accumulates the head of the stream until the hostname can be read out of
     * it. Returns true when the bytes were consumed by the sniffer and must not
     * be forwarded yet.
     */
    private fun bufferForHostCheck(packet: ByteArray, offset: Int, length: Int): Boolean {
        var buf = sniffBuffer
        if (buf == null) {
            buf = ByteArray(SNIFF_LIMIT)
            sniffBuffer = buf
        }
        val room = SNIFF_LIMIT - sniffLength
        val copyLen = min(room, length)
        System.arraycopy(packet, offset, buf, sniffLength, copyLen)
        sniffLength += copyLen

        val result = HostSniffer.sniff(buf, sniffLength)
        val overflowed = sniffLength >= SNIFF_LIMIT
        if (result is HostSniffer.Result.NeedMore && !overflowed) return true

        hostResolved = true
        val host = (result as? HostSniffer.Result.Host)?.name
        if (host != null && Blocklist.isBlocked(host)) {
            Stats.blocked.incrementAndGet()
            Log.d(TAG, "blocked $host ($key)")
            sendReset()
            closeHard()
            return true
        }
        engine.onHostSeen(key, host)

        // Release everything we withheld, plus whatever did not fit.
        enqueueToRemote(ByteBuffer.wrap(buf, 0, sniffLength))
        if (copyLen < length) {
            val rest = ByteArray(length - copyLen)
            System.arraycopy(packet, offset + copyLen, rest, 0, rest.size)
            enqueueToRemote(ByteBuffer.wrap(rest))
        }
        sniffBuffer = null
        sniffLength = 0
        return true
    }

    private fun enqueueToRemote(buffer: ByteBuffer) {
        pendingToRemoteBytes += buffer.remaining()
        pendingToRemote.addLast(buffer)
        flushToRemote()
    }

    private fun onAppFin() {
        appFinReceived = true
        shutdownAfterFlush = true
        flushToRemote()
        maybeFinish()
    }

    private fun onAck(ackNum: Long) {
        if (seqDiff(ackNum, sendUnack) <= 0 || seqDiff(ackNum, sendSeq) > 0) return
        sendUnack = ackNum
        while (unacked.isNotEmpty()) {
            val seg = unacked.first()
            if (seqDiff(seqAdd(seg.seq, seg.seqLength.toLong()), ackNum) <= 0) {
                unacked.removeFirst()
            } else {
                break
            }
        }
        maybeFinish()
    }

    // -- us -> remote --------------------------------------------------------

    fun onWritable() {
        flushToRemote()
        updateInterest()
    }

    private fun flushToRemote() {
        val ch = channel ?: return
        waitingForUploadTokens = false
        try {
            while (pendingToRemote.isNotEmpty()) {
                val buf = pendingToRemote.first()
                val grant = engine.upBucket.take(buf.remaining())
                if (grant <= 0) {
                    waitingForUploadTokens = true
                    break
                }

                val savedLimit = buf.limit()
                buf.limit(buf.position() + grant)
                val written = ch.write(buf)
                buf.limit(savedLimit)

                if (written < grant) engine.upBucket.refund(grant - written)
                pendingToRemoteBytes -= written
                if (!buf.hasRemaining()) pendingToRemote.removeFirst()
                if (written == 0) break
            }
        } catch (e: IOException) {
            Log.d(TAG, "write failed for $key: ${e.message}")
            sendReset()
            closeHard()
            return
        }

        if (shutdownAfterFlush && pendingToRemote.isEmpty()) {
            shutdownAfterFlush = false
            try {
                ch.shutdownOutput()
            } catch (_: IOException) {
                // Peer already gone; the read side will surface it.
            }
        }
    }

    // -- remote -> app -------------------------------------------------------

    fun onReadable() {
        val ch = channel ?: return
        if (state != State.ESTABLISHED || finSentToApp) {
            updateInterest()
            return
        }

        val space = windowAvailable()
        if (space <= 0) {
            updateInterest()
            return
        }

        val grant = engine.downBucket.take(min(space, READ_CHUNK))
        if (grant <= 0) {
            updateInterest()
            return
        }

        val buffer = ByteArray(grant)
        val read = try {
            ch.read(ByteBuffer.wrap(buffer))
        } catch (e: IOException) {
            engine.downBucket.refund(grant)
            Log.d(TAG, "read failed for $key: ${e.message}")
            sendReset()
            closeHard()
            return
        }

        if (read < 0) {
            engine.downBucket.refund(grant)
            onRemoteEof()
            return
        }
        if (read < grant) engine.downBucket.refund(grant - read)
        if (read == 0) {
            updateInterest()
            return
        }

        Stats.addDown(read)
        lastActivity = SystemClock.elapsedRealtime()
        emitToApp(buffer, read)
        updateInterest()
    }

    private fun emitToApp(data: ByteArray, length: Int) {
        var offset = 0
        while (offset < length) {
            val chunk = min(Proto.MSS, length - offset)
            val payload = ByteArray(chunk)
            System.arraycopy(data, offset, payload, 0, chunk)
            transmit(Proto.PSH or Proto.ACK, payload, chunk)
            offset += chunk
        }
    }

    private fun onRemoteEof() {
        if (finSentToApp) return
        transmit(Proto.FIN or Proto.ACK, null, 1)
        finSentToApp = true
        closingSince = SystemClock.elapsedRealtime()
        updateInterest()
        maybeFinish()
    }

    // -- packet emission -----------------------------------------------------

    private fun sendSynAck() {
        val seg = Segment(sendSeq, Proto.SYN or Proto.ACK, null, 1, SystemClock.elapsedRealtime(), 0)
        unacked.addLast(seg)
        sendSeq = seqAdd(sendSeq, 1)
        engine.transmitToTun(buildPacket(seg))
    }

    /** Queues a segment for the app and remembers it for retransmission. */
    private fun transmit(flags: Int, payload: ByteArray?, seqLength: Int) {
        val seg = Segment(sendSeq, flags, payload, seqLength, SystemClock.elapsedRealtime(), 0)
        unacked.addLast(seg)
        sendSeq = seqAdd(sendSeq, seqLength.toLong())
        engine.transmitToTun(buildPacket(seg))
    }

    private fun buildPacket(seg: Segment): ByteArray = PacketBuilder.tcp(
        srcIp = key.dstIp,
        srcPort = key.dstPort,
        dstIp = key.srcIp,
        dstPort = key.srcPort,
        seq = seg.seq,
        ack = recvNext,
        flags = seg.flags,
        window = advertisedWindow(),
        payload = seg.payload,
        payloadLength = seg.payload?.size ?: 0,
        mssOption = if (seg.flags and Proto.SYN != 0) Proto.MSS else 0,
    )

    private fun sendAck() {
        engine.transmitToTun(
            PacketBuilder.tcp(
                srcIp = key.dstIp,
                srcPort = key.dstPort,
                dstIp = key.srcIp,
                dstPort = key.srcPort,
                seq = sendSeq,
                ack = recvNext,
                flags = Proto.ACK,
                window = advertisedWindow(),
            )
        )
    }

    fun sendReset() {
        engine.transmitToTun(
            PacketBuilder.tcp(
                srcIp = key.dstIp,
                srcPort = key.dstPort,
                dstIp = key.srcIp,
                dstPort = key.srcPort,
                seq = sendSeq,
                ack = recvNext,
                flags = Proto.RST or Proto.ACK,
                window = 0,
            )
        )
    }

    // -- housekeeping --------------------------------------------------------

    /** Called on the engine tick. Returns false when the session may be dropped. */
    fun tick(now: Long): Boolean {
        if (state == State.CLOSED) return false

        if (state == State.CONNECTING && now - lastActivity > CONNECT_TIMEOUT_MS) {
            Log.d(TAG, "connect timeout for $key")
            sendReset()
            closeHard()
            return false
        }

        if (pendingToRemote.isNotEmpty()) flushToRemote()
        retransmitOldest(force = false)

        val idleLimit = if (finSentToApp || appFinReceived) CLOSING_TIMEOUT_MS else IDLE_TIMEOUT_MS
        if (now - lastActivity > idleLimit) {
            sendReset()
            closeHard()
            return false
        }

        if (closingSince != 0L && appFinReceived && now - closingSince > LINGER_MS) {
            closeHard()
            return false
        }

        updateInterest()
        return state != State.CLOSED
    }

    private fun retransmitOldest(force: Boolean) {
        val seg = unacked.firstOrNull() ?: return
        val now = SystemClock.elapsedRealtime()
        val backoff = RTO_MS shl min(seg.tries, 4)
        if (!force && now - seg.sentAt < backoff) return
        if (seg.tries >= MAX_RETRIES) {
            Log.d(TAG, "giving up retransmitting $key")
            sendReset()
            closeHard()
            return
        }
        seg.tries++
        seg.sentAt = now
        engine.transmitToTun(buildPacket(seg))
    }

    private fun maybeFinish() {
        if (finSentToApp && appFinReceived && unacked.isEmpty()) closeHard()
    }

    private fun windowAvailable(): Int {
        val inFlight = seqDiff(sendSeq, sendUnack).toInt()
        return (min(appWindow, MAX_IN_FLIGHT) - inFlight).coerceAtLeast(0)
    }

    private fun advertisedWindow(): Int =
        (RECV_BUFFER - pendingToRemoteBytes).coerceIn(0, 65535)

    private fun updateInterest() {
        val k = selectionKey ?: return
        if (!k.isValid) return
        val ops = when {
            state == State.CONNECTING -> SelectionKey.OP_CONNECT
            state == State.CLOSED -> 0
            else -> {
                var o = 0
                if (canRead()) o = o or SelectionKey.OP_READ
                if (pendingToRemote.isNotEmpty() && !waitingForUploadTokens) o = o or SelectionKey.OP_WRITE
                o
            }
        }
        if (k.interestOps() != ops) k.interestOps(ops)
    }

    private fun canRead(): Boolean =
        !finSentToApp && windowAvailable() > 0 && engine.downBucket.available() > 0

    fun closeHard() {
        if (state == State.CLOSED) return
        state = State.CLOSED
        try {
            selectionKey?.cancel()
        } catch (_: Exception) {
        }
        try {
            channel?.close()
        } catch (_: IOException) {
        }
        channel = null
        selectionKey = null
        pendingToRemote.clear()
        pendingToRemoteBytes = 0
        unacked.clear()
        sniffBuffer = null
        engine.onSessionClosed(this)
    }

    private companion object {
        const val TAG = "TikSaver/tcp"

        /** Ceiling on unacknowledged bytes towards the app. */
        const val MAX_IN_FLIGHT = 64 * 1024

        /** How much we are willing to buffer from the app before shrinking the window. */
        const val RECV_BUFFER = 64 * 1024

        const val READ_CHUNK = 32 * 1024
        const val SNIFF_LIMIT = 4096

        const val RTO_MS = 300L
        const val MAX_RETRIES = 6
        const val CONNECT_TIMEOUT_MS = 12_000L
        const val IDLE_TIMEOUT_MS = 600_000L
        const val CLOSING_TIMEOUT_MS = 30_000L
        const val LINGER_MS = 5_000L
    }
}
