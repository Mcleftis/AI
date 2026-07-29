package com.leftis.tiksaver.net

import java.net.InetAddress
import java.util.concurrent.atomic.AtomicLong

/** Immutable snapshot of the user's settings handed to the packet engine. */
internal data class EngineConfig(
    val downKbps: Int,
    val upKbps: Int,
    val blockQuic: Boolean,
    val blockAds: Boolean,
    val upstreamDns: List<InetAddress>,
    /** Virtual resolver advertised inside the tunnel, as a raw IPv4 int. */
    val virtualDns: Int,
) {
    val downBytesPerSecond: Long get() = downKbps * 1000L / 8L
    val upBytesPerSecond: Long get() = upKbps * 1000L / 8L
}

/** Live counters, read by the UI and the notification. */
internal object Stats {
    val bytesDown = AtomicLong()
    val bytesUp = AtomicLong()
    val blocked = AtomicLong()
    val openConnections = AtomicLong()
    val startedAt = AtomicLong()

    /** Bytes seen in the current sampling window, used to show a live rate. */
    private val windowBytes = AtomicLong()
    private val windowStart = AtomicLong()

    @Volatile
    var lastRateBytesPerSecond: Long = 0
        private set

    fun reset() {
        bytesDown.set(0)
        bytesUp.set(0)
        blocked.set(0)
        windowBytes.set(0)
        windowStart.set(android.os.SystemClock.elapsedRealtime())
        startedAt.set(android.os.SystemClock.elapsedRealtime())
        lastRateBytesPerSecond = 0
    }

    fun addDown(n: Int) {
        bytesDown.addAndGet(n.toLong())
        sample(n)
    }

    fun addUp(n: Int) {
        bytesUp.addAndGet(n.toLong())
        sample(n)
    }

    private fun sample(n: Int) {
        val now = android.os.SystemClock.elapsedRealtime()
        val start = windowStart.get()
        val total = windowBytes.addAndGet(n.toLong())
        val elapsed = now - start
        if (elapsed >= 1000) {
            if (windowStart.compareAndSet(start, now)) {
                lastRateBytesPerSecond = total * 1000 / elapsed
                windowBytes.addAndGet(-total)
            }
        }
    }

    /** Called on the UI tick so the rate decays to zero when traffic stops. */
    fun decayIfIdle() {
        val now = android.os.SystemClock.elapsedRealtime()
        if (now - windowStart.get() >= 2500) {
            lastRateBytesPerSecond = 0
        }
    }
}
