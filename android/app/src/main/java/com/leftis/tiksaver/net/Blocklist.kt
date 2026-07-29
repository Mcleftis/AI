package com.leftis.tiksaver.net

/**
 * Hostname filter for advertising, attribution and telemetry endpoints.
 *
 * The list is deliberately conservative. ByteDance serves the API, the video
 * CDN and the log collectors from the same registrable domains, so blanket
 * blocking `tiktokv.com` would simply break the app. Instead we block whole
 * domains only when the domain exists purely to serve ads, and otherwise match
 * the well-known telemetry hostname prefixes under the ByteDance base domains.
 */
internal object Blocklist {

    /** Whole domains that only ever carry ads, attribution or measurement. */
    private val blockedDomains = setOf(
        // ByteDance / Pangle advertising stack
        "pangle.io",
        "pangle-sg.io",
        "pangolin-sdk-toutiao.com",
        "pangolin-sdk-toutiao-b.com",
        "pglstatp-toutiao.com",
        "ads.tiktok.com",
        "ads-sg.tiktok.com",
        "ads-api.tiktok.com",
        "analytics.tiktok.com",
        "analytics-sg.tiktok.com",
        "business-api.tiktok.com",
        // Third-party ad networks and attribution SDKs commonly bundled in apps
        "doubleclick.net",
        "googleadservices.com",
        "googlesyndication.com",
        "adservice.google.com",
        "app-measurement.com",
        "appsflyer.com",
        "appsflyersdk.com",
        "adjust.com",
        "adjust.io",
        "branch.io",
        "kochava.com",
        "singular.net",
        "unityads.unity3d.com",
        "applovin.com",
    )

    /** ByteDance base domains that mix product traffic with telemetry traffic. */
    private val telemetryBaseDomains = setOf(
        "tiktokv.com",
        "tiktokv.us",
        "tiktokv.eu",
        "tiktokv.co",
        "tiktokw.us",
        "byteoversea.com",
        "bytedance.com",
        "bytedanceapi.com",
        "snssdk.com",
        "isnssdk.com",
        "musical.ly",
    )

    /**
     * Leading label of a telemetry host, e.g. `log16-normal-c-useast1a`,
     * `mon-va`, `mcs`. The boundary check keeps `login`, `monitorless` and
     * friends out of the match.
     */
    private val telemetryPrefixes = listOf(
        "log", "mon", "mcs", "applog", "ads", "analytics", "track", "report", "sentry", "crash",
    )

    fun isBlocked(hostname: String?): Boolean {
        if (hostname.isNullOrEmpty()) return false
        val host = hostname.lowercase().trimEnd('.')
        if (host.isEmpty()) return false

        if (matchesDomain(host, blockedDomains)) return true

        val firstLabel = host.substringBefore('.')
        val base = telemetryBaseDomains.firstOrNull { host == it || host.endsWith(".$it") }
            ?: return false
        // Never filter the host that *is* the base domain itself.
        if (host == base) return false
        return telemetryPrefixes.any { hasLabelPrefix(firstLabel, it) }
    }

    private fun matchesDomain(host: String, domains: Set<String>): Boolean =
        domains.any { host == it || host.endsWith(".$it") }

    /**
     * True when [label] starts with [prefix] and the next character is a digit
     * or a separator, so the prefix really is a word and not the start of a
     * longer one.
     */
    private fun hasLabelPrefix(label: String, prefix: String): Boolean {
        if (!label.startsWith(prefix)) return false
        if (label.length == prefix.length) return true
        val next = label[prefix.length]
        return next.isDigit() || next == '-' || next == '_'
    }
}
