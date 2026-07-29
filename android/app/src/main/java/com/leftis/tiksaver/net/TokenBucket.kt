package com.leftis.tiksaver.net

import android.os.SystemClock

/**
 * Byte-granular token bucket used to pace one direction of the tunnel.
 *
 * We never drop packets to enforce the rate. Instead the bucket limits how many
 * bytes we are willing to read out of a remote socket per unit of time; TCP's
 * own flow control then squeezes the sender. That is what makes TikTok's
 * adaptive bitrate logic settle on a low-quality rendition instead of just
 * retransmitting a high-quality one.
 */
internal class TokenBucket(bytesPerSecond: Long) {

    @Volatile
    private var rate: Long = bytesPerSecond.coerceAtLeast(MIN_RATE)

    @Volatile
    private var capacity: Long = burstFor(bytesPerSecond)

    private var tokens: Double = capacity.toDouble()
    private var lastRefillNanos: Long = SystemClock.elapsedRealtimeNanos()

    fun setRate(bytesPerSecond: Long) = synchronized(this) {
        refill()
        rate = bytesPerSecond.coerceAtLeast(MIN_RATE)
        capacity = burstFor(rate)
        if (tokens > capacity) tokens = capacity.toDouble()
    }

    private fun refill() {
        val now = SystemClock.elapsedRealtimeNanos()
        val elapsed = now - lastRefillNanos
        if (elapsed <= 0) return
        lastRefillNanos = now
        tokens = (tokens + rate * (elapsed / 1_000_000_000.0)).coerceAtMost(capacity.toDouble())
    }

    /** Grants up to [max] bytes, returning how many are actually allowed now. */
    fun take(max: Int): Int = synchronized(this) {
        if (max <= 0) return 0
        refill()
        val granted = minOf(max.toLong(), tokens.toLong()).toInt()
        if (granted > 0) tokens -= granted
        return granted
    }

    /** Returns tokens granted by [take] that ended up unused. */
    fun refund(bytes: Int) = synchronized(this) {
        if (bytes <= 0) return
        tokens = (tokens + bytes).coerceAtMost(capacity.toDouble())
    }

    /** Bytes currently available without waiting; used to re-arm throttled reads. */
    fun available(): Int = synchronized(this) {
        refill()
        return tokens.coerceAtMost(Int.MAX_VALUE.toDouble()).toInt()
    }

    private companion object {
        /** Below ~2 KB/s nothing usable gets through, and we would divide by ~0. */
        const val MIN_RATE = 2048L

        /**
         * A quarter second of burst keeps small API calls and thumbnails snappy
         * while still being far too small to fit a high-bitrate video chunk.
         */
        fun burstFor(rate: Long): Long = (rate / 4).coerceIn(8L * Proto.MSS, 512L * 1024L)
    }
}
