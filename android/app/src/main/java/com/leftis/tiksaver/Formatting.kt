package com.leftis.tiksaver

import java.util.Locale

/** Human-readable byte counts, always in decimal units to match carrier billing. */
fun formatBytes(bytes: Long): String {
    val b = bytes.coerceAtLeast(0)
    return when {
        b < 1_000 -> "$b B"
        b < 1_000_000 -> String.format(Locale.US, "%.1f kB", b / 1_000.0)
        b < 1_000_000_000 -> String.format(Locale.US, "%.1f MB", b / 1_000_000.0)
        else -> String.format(Locale.US, "%.2f GB", b / 1_000_000_000.0)
    }
}

fun formatRate(bytesPerSecond: Long): String =
    if (bytesPerSecond < 1_000) "$bytesPerSecond B/s"
    else String.format(Locale.US, "%.0f kB/s", bytesPerSecond / 1_000.0)

/** The exact ceiling a given cap implies, which is the honest way to state savings. */
fun megabytesPerMinute(kbps: Int): String {
    val megabytes = kbps * 60.0 / 8.0 / 1000.0
    return String.format(Locale.US, "%.1f MB", megabytes)
}
