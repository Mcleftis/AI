package com.leftis.tiksaver.net

import android.os.ParcelFileDescriptor
import android.os.SystemClock
import android.util.Log
import java.io.FileInputStream
import java.io.FileOutputStream
import java.io.IOException
import java.net.DatagramSocket
import java.net.InetAddress
import java.net.InetSocketAddress
import java.net.Socket
import java.nio.channels.CancelledKeyException
import java.nio.channels.SelectionKey
import java.nio.channels.Selector
import java.util.concurrent.ArrayBlockingQueue
import java.util.concurrent.LinkedBlockingQueue

/** Lets the engine keep its own sockets outside the tunnel it is servicing. */
interface SocketProtector {
    fun protect(socket: Socket): Boolean
    fun protect(socket: DatagramSocket): Boolean
}

/**
 * The packet engine.
 *
 * Three threads and one rule: every piece of session state is touched only by
 * the worker thread, so nothing in [TcpSession] or [UdpSession] needs locking.
 *
 *  * reader  — blocking reads from the tun fd into [inbound]
 *  * worker  — drains [inbound], runs the selector, owns all session state
 *  * writer  — drains [outbound] onto the tun fd
 */
internal class NatEngine(
    private val tun: ParcelFileDescriptor,
    initialConfig: EngineConfig,
    val protector: SocketProtector,
) {

    @Volatile
    var config: EngineConfig = initialConfig
        private set

    /** When true every new flow is refused; used for the screen-off blackout. */
    @Volatile
    var blackout: Boolean = false

    val selector: Selector = Selector.open()
    val downBucket = TokenBucket(initialConfig.downBytesPerSecond)
    val upBucket = TokenBucket(initialConfig.upBytesPerSecond)

    private val tcpSessions = HashMap<FlowKey, TcpSession>()
    private val udpSessions = HashMap<FlowKey, UdpSession>()

    private val inbound = ArrayBlockingQueue<ByteArray>(1024)
    private val outbound = LinkedBlockingQueue<ByteArray>(2048)

    @Volatile
    private var running = false

    private var readerThread: Thread? = null
    private var writerThread: Thread? = null
    private var workerThread: Thread? = null

    private var lastTick = 0L

    fun updateConfig(newConfig: EngineConfig) {
        config = newConfig
        downBucket.setRate(newConfig.downBytesPerSecond)
        upBucket.setRate(newConfig.upBytesPerSecond)
    }

    fun start() {
        if (running) return
        running = true

        readerThread = Thread({ readLoop() }, "tiksaver-reader").apply { start() }
        writerThread = Thread({ writeLoop() }, "tiksaver-writer").apply { start() }
        workerThread = Thread({ workerLoop() }, "tiksaver-worker").apply { start() }
    }

    fun stop() {
        if (!running) return
        running = false
        selector.wakeup()
        outbound.offer(POISON)
        try {
            tun.close()
        } catch (_: IOException) {
        }
        workerThread?.join(2000)
        readerThread?.interrupt()
        writerThread?.interrupt()
        try {
            selector.close()
        } catch (_: IOException) {
        }
        Stats.openConnections.set(0)
    }

    // -- tun I/O -------------------------------------------------------------

    private fun readLoop() {
        val input = FileInputStream(tun.fileDescriptor)
        // Comfortably larger than the MTU: a short read would silently truncate
        // a packet, and a truncated packet is a corrupt one.
        val scratch = ByteArray(READ_BUFFER)
        try {
            while (running) {
                val n = input.read(scratch)
                if (n <= 0) {
                    if (n < 0) break
                    continue
                }
                if (!inbound.offer(scratch.copyOf(n))) {
                    // Worker is behind. Dropping is the correct back-pressure
                    // signal here; TCP will retransmit.
                    continue
                }
                selector.wakeup()
            }
        } catch (e: IOException) {
            if (running) Log.w(TAG, "tun read ended: ${e.message}")
        } finally {
            running = false
            selector.wakeup()
        }
    }

    private fun writeLoop() {
        val output = FileOutputStream(tun.fileDescriptor)
        try {
            while (running) {
                val packet = outbound.take()
                if (packet === POISON) break
                output.write(packet)
            }
        } catch (_: InterruptedException) {
            Thread.currentThread().interrupt()
        } catch (e: IOException) {
            if (running) Log.w(TAG, "tun write ended: ${e.message}")
        }
    }

    fun transmitToTun(packet: ByteArray) {
        if (!outbound.offer(packet)) Log.d(TAG, "outbound queue full, dropping packet")
    }

    // -- worker --------------------------------------------------------------

    private fun workerLoop() {
        try {
            while (running) {
                drainInbound()
                if (!running) break

                try {
                    selector.select(selectTimeout())
                } catch (e: IOException) {
                    if (running) Log.w(TAG, "select failed: ${e.message}")
                    break
                }

                val iterator = selector.selectedKeys().iterator()
                while (iterator.hasNext()) {
                    val key = iterator.next()
                    iterator.remove()
                    dispatch(key)
                }

                housekeeping()
            }
        } catch (e: Throwable) {
            Log.e(TAG, "worker crashed", e)
        } finally {
            closeAllSessions()
        }
    }

    private fun dispatch(key: SelectionKey) {
        try {
            if (!key.isValid) return
            when (val session = key.attachment()) {
                is TcpSession -> {
                    if (key.isConnectable) {
                        session.onConnectable()
                        return
                    }
                    if (key.isReadable) session.onReadable()
                    if (key.isValid && key.isWritable) session.onWritable()
                }

                is UdpSession -> if (key.isReadable) session.onReadable()
            }
        } catch (_: CancelledKeyException) {
            // Session closed itself while we were handling it.
        } catch (e: Exception) {
            Log.w(TAG, "dispatch failed", e)
            (key.attachment() as? TcpSession)?.closeHard()
            (key.attachment() as? UdpSession)?.close()
        }
    }

    private fun selectTimeout(): Long = when {
        tcpSessions.isEmpty() && udpSessions.isEmpty() -> 1000L
        downBucket.available() < Proto.MSS -> 15L
        else -> 100L
    }

    private fun housekeeping() {
        val now = SystemClock.elapsedRealtime()
        if (now - lastTick < TICK_MS) return
        lastTick = now

        // A blackout has to take effect on flows that are already running,
        // otherwise a download started before the screen went off keeps going.
        if (blackout && (tcpSessions.isNotEmpty() || udpSessions.isNotEmpty())) {
            for (session in tcpSessions.values.toList()) session.sendReset()
            closeAllSessions()
            Stats.openConnections.set(0)
            return
        }

        if (tcpSessions.isNotEmpty()) {
            for (session in tcpSessions.values.toList()) session.tick(now)
        }
        if (udpSessions.isNotEmpty()) {
            for (session in udpSessions.values.toList()) session.tick(now)
        }
        Stats.openConnections.set((tcpSessions.size + udpSessions.size).toLong())
    }

    private fun closeAllSessions() {
        for (session in tcpSessions.values.toList()) session.closeHard()
        for (session in udpSessions.values.toList()) session.close()
        tcpSessions.clear()
        udpSessions.clear()
    }

    // -- packet handling -----------------------------------------------------

    private fun drainInbound() {
        var handled = 0
        while (handled < MAX_BATCH) {
            val packet = inbound.poll() ?: return
            handle(packet)
            handled++
        }
    }

    private fun handle(data: ByteArray) {
        if (data.size < Proto.IP_HDR) return
        val ip = Ipv4Packet(data, data.size)

        // IPv6 is routed into the tunnel on purpose and dropped here, so the app
        // falls back to IPv4 rather than escaping the shaper over a second
        // address family we do not terminate.
        if (ip.version != 4) return
        if (ip.headerLength < Proto.IP_HDR || ip.headerLength > data.size) return
        if (ip.isFragment) return

        val payloadLength = ip.payloadLength
        if (payloadLength <= 0) return

        when (ip.protocol) {
            Proto.TCP -> handleTcp(ip, payloadLength)
            Proto.UDP -> handleUdp(ip, payloadLength)
            else -> Unit
        }
    }

    private fun handleTcp(ip: Ipv4Packet, transportLength: Int) {
        if (transportLength < Proto.TCP_HDR) return
        val t = ip.payloadOffset
        val srcPort = ip.data.u16(t)
        val dstPort = ip.data.u16(t + 2)
        val flags = ip.data.u8(t + 13)

        val key = FlowKey(Proto.TCP, ip.srcIp, srcPort, ip.dstIp, dstPort)
        val isSyn = flags and Proto.SYN != 0 && flags and Proto.ACK == 0

        val existing = tcpSessions[key]
        if (existing != null) {
            // The app can reuse a source port. A SYN whose sequence number is
            // not the one that opened this session is a genuinely new
            // connection, not a retransmission, so retire the old one.
            if (!isSyn || existing.isSynRetransmit(ip.data.u32(t + 4))) {
                existing.onPacket(ip.data, t, transportLength)
                return
            }
            existing.closeHard()
        }

        if (!isSyn) {
            // Nothing here knows about this flow; tell the app to give up
            // instead of letting it retransmit into the void.
            if (flags and Proto.RST == 0) sendStrayReset(ip, t)
            return
        }

        if (blackout) {
            Stats.blocked.incrementAndGet()
            sendStrayReset(ip, t)
            return
        }

        if (tcpSessions.size >= MAX_TCP_SESSIONS) {
            Log.w(TAG, "session table full, refusing $key")
            sendStrayReset(ip, t)
            return
        }

        val seq = ip.data.u32(t + 4)
        val window = ip.data.u16(t + 14)
        val session = TcpSession(key, this)
        tcpSessions[key] = session
        if (!session.open(seq, window)) {
            tcpSessions.remove(key)
            sendStrayReset(ip, t)
        }
    }

    /** RST for a flow we have no session for, built straight from the trigger. */
    private fun sendStrayReset(ip: Ipv4Packet, transportOffset: Int) {
        val t = transportOffset
        val srcPort = ip.data.u16(t)
        val dstPort = ip.data.u16(t + 2)
        val seq = ip.data.u32(t + 4)
        val dataOffset = (ip.data.u8(t + 12) ushr 4) * 4
        val flags = ip.data.u8(t + 13)
        val payload = (ip.payloadLength - dataOffset).coerceAtLeast(0)
        var consumed = payload.toLong()
        if (flags and Proto.SYN != 0) consumed += 1
        if (flags and Proto.FIN != 0) consumed += 1

        transmitToTun(
            PacketBuilder.tcp(
                srcIp = ip.dstIp,
                srcPort = dstPort,
                dstIp = ip.srcIp,
                dstPort = srcPort,
                seq = 0,
                ack = seqAdd(seq, consumed),
                flags = Proto.RST or Proto.ACK,
                window = 0,
            )
        )
    }

    private fun handleUdp(ip: Ipv4Packet, transportLength: Int) {
        if (transportLength < Proto.UDP_HDR) return
        val u = ip.payloadOffset
        val srcPort = ip.data.u16(u)
        val dstPort = ip.data.u16(u + 2)

        val declaredLength = ip.data.u16(u + 4)
        val payloadOffset = u + Proto.UDP_HDR
        val payloadLength = minOf(declaredLength - Proto.UDP_HDR, transportLength - Proto.UDP_HDR)
        if (payloadLength < 0) return

        if (blackout) {
            Stats.blocked.incrementAndGet()
            return
        }

        if (dstPort == PORT_QUIC && config.blockQuic) {
            Stats.blocked.incrementAndGet()
            PacketBuilder.icmpPortUnreachable(ip)?.let { transmitToTun(it) }
            return
        }

        val key = FlowKey(Proto.UDP, ip.srcIp, srcPort, ip.dstIp, dstPort)

        if (dstPort == PORT_DNS) {
            handleDns(ip, key, payloadOffset, payloadLength)
            return
        }

        val session = udpSessions[key] ?: run {
            if (udpSessions.size >= MAX_UDP_SESSIONS) return
            val target = InetSocketAddress(InetAddress.getByAddress(ipToBytes(ip.dstIp)), dstPort)
            val created = UdpSession(key, target, this, shaped = true)
            if (!created.open()) return
            udpSessions[key] = created
            created
        }
        session.send(ip.data, payloadOffset, payloadLength)
    }

    private fun handleDns(ip: Ipv4Packet, key: FlowKey, payloadOffset: Int, payloadLength: Int) {
        if (payloadLength <= 0) return

        if (config.blockAds) {
            val query = ByteArray(payloadLength)
            System.arraycopy(ip.data, payloadOffset, query, 0, payloadLength)
            val name = DnsMessage.questionName(query, payloadLength)
            if (Blocklist.isBlocked(name)) {
                Stats.blocked.incrementAndGet()
                DnsMessage.nxdomain(query, payloadLength)?.let { answer ->
                    transmitToTun(
                        PacketBuilder.udp(
                            srcIp = key.dstIp,
                            srcPort = key.dstPort,
                            dstIp = key.srcIp,
                            dstPort = key.srcPort,
                            payload = answer,
                        )
                    )
                }
                return
            }
        }

        val session = udpSessions[key] ?: run {
            if (udpSessions.size >= MAX_UDP_SESSIONS) return
            val upstream = resolveUpstream(ip.dstIp) ?: return
            val created = UdpSession(key, upstream, this, shaped = false)
            if (!created.open()) return
            udpSessions[key] = created
            created
        }
        session.send(ip.data, payloadOffset, payloadLength)
    }

    /**
     * Queries aimed at the virtual resolver we advertise inside the tunnel are
     * relayed to a real one; anything else keeps its original destination.
     */
    private fun resolveUpstream(destination: Int): InetSocketAddress? = try {
        if (destination == config.virtualDns) {
            config.upstreamDns.firstOrNull()?.let { InetSocketAddress(it, PORT_DNS) }
        } else {
            InetSocketAddress(InetAddress.getByAddress(ipToBytes(destination)), PORT_DNS)
        }
    } catch (e: Exception) {
        Log.w(TAG, "no upstream resolver: ${e.message}")
        null
    }

    // -- session callbacks ---------------------------------------------------

    fun onSessionClosed(session: TcpSession) {
        if (tcpSessions[session.key] === session) tcpSessions.remove(session.key)
    }

    fun onSessionClosed(session: UdpSession) {
        if (udpSessions[session.key] === session) udpSessions.remove(session.key)
    }

    fun onHostSeen(key: FlowKey, host: String?) {
        if (host != null) Log.v(TAG, "$key -> $host")
    }

    private companion object {
        const val TAG = "TikSaver/engine"
        const val READ_BUFFER = 32 * 1024
        const val TICK_MS = 50L
        const val MAX_BATCH = 256
        const val MAX_TCP_SESSIONS = 512
        const val MAX_UDP_SESSIONS = 256
        const val PORT_DNS = 53
        const val PORT_QUIC = 443
        val POISON = ByteArray(0)
    }
}
