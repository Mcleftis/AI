package com.leftis.tiksaver.net

import android.os.SystemClock
import android.util.Log
import java.io.IOException
import java.net.InetSocketAddress
import java.nio.ByteBuffer
import java.nio.channels.DatagramChannel
import java.nio.channels.SelectionKey

/**
 * A UDP "connection", which for us is just a 5-tuple with an idle timer.
 *
 * Unlike TCP there is no flow control to lean on, so shaping here means
 * dropping datagrams once the bucket is empty. That is the right signal for the
 * congestion-controlled protocols that actually matter over UDP (QUIC): loss
 * makes them back off, exactly like a slow link would.
 *
 * DNS is exempt from shaping — starving it would only add latency to every
 * request without saving a meaningful number of bytes.
 */
internal class UdpSession(
    val key: FlowKey,
    private val remote: InetSocketAddress,
    private val engine: NatEngine,
    private val shaped: Boolean,
) {

    private var channel: DatagramChannel? = null
    private var selectionKey: SelectionKey? = null
    private var lastActivity = SystemClock.elapsedRealtime()

    fun open(): Boolean = try {
        val ch = DatagramChannel.open()
        ch.configureBlocking(false)
        if (!engine.protector.protect(ch.socket())) {
            ch.close()
            false
        } else {
            ch.connect(remote)
            channel = ch
            selectionKey = ch.register(engine.selector, SelectionKey.OP_READ, this)
            true
        }
    } catch (e: Exception) {
        Log.w(TAG, "udp open failed for $key: ${e.message}")
        close()
        false
    }

    fun send(packet: ByteArray, offset: Int, length: Int) {
        val ch = channel ?: return
        lastActivity = SystemClock.elapsedRealtime()
        Stats.addUp(length)
        if (shaped) {
            val grant = engine.upBucket.take(length)
            if (grant < length) {
                engine.upBucket.refund(grant)
                return
            }
        }
        try {
            ch.write(ByteBuffer.wrap(packet, offset, length))
        } catch (e: IOException) {
            Log.d(TAG, "udp write failed for $key: ${e.message}")
            close()
        }
    }

    fun onReadable() {
        val ch = channel ?: return
        val buffer = ByteArray(MAX_DATAGRAM)
        while (true) {
            val read = try {
                ch.read(ByteBuffer.wrap(buffer))
            } catch (e: IOException) {
                Log.d(TAG, "udp read failed for $key: ${e.message}")
                close()
                return
            }
            if (read <= 0) return

            lastActivity = SystemClock.elapsedRealtime()
            if (shaped) {
                val grant = engine.downBucket.take(read)
                if (grant < read) {
                    engine.downBucket.refund(grant)
                    continue // dropped on purpose: this is the shaping signal
                }
            }
            Stats.addDown(read)
            engine.transmitToTun(
                PacketBuilder.udp(
                    srcIp = key.dstIp,
                    srcPort = key.dstPort,
                    dstIp = key.srcIp,
                    dstPort = key.srcPort,
                    payload = buffer,
                    payloadOffset = 0,
                    payloadLength = read,
                )
            )
        }
    }

    fun tick(now: Long): Boolean {
        val limit = if (shaped) IDLE_TIMEOUT_MS else DNS_TIMEOUT_MS
        if (now - lastActivity > limit) {
            close()
            return false
        }
        return true
    }

    fun close() {
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
        engine.onSessionClosed(this)
    }

    private companion object {
        const val TAG = "TikSaver/udp"
        const val MAX_DATAGRAM = 32 * 1024
        const val IDLE_TIMEOUT_MS = 60_000L
        const val DNS_TIMEOUT_MS = 15_000L
    }
}
