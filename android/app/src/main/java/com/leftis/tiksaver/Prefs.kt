package com.leftis.tiksaver

import android.content.Context
import android.content.pm.PackageManager

/** Everything the user can change, persisted in one SharedPreferences file. */
class Prefs(context: Context) {

    private val store = context.applicationContext
        .getSharedPreferences("tiksaver", Context.MODE_PRIVATE)

    var downKbps: Int
        get() = store.getInt(KEY_DOWN, DEFAULT_DOWN_KBPS)
        set(value) = store.edit().putInt(KEY_DOWN, value).apply()

    var upKbps: Int
        get() = store.getInt(KEY_UP, DEFAULT_UP_KBPS)
        set(value) = store.edit().putInt(KEY_UP, value).apply()

    var cellularOnly: Boolean
        get() = store.getBoolean(KEY_CELLULAR_ONLY, true)
        set(value) = store.edit().putBoolean(KEY_CELLULAR_ONLY, value).apply()

    var blockQuic: Boolean
        get() = store.getBoolean(KEY_BLOCK_QUIC, true)
        set(value) = store.edit().putBoolean(KEY_BLOCK_QUIC, value).apply()

    var blockAds: Boolean
        get() = store.getBoolean(KEY_BLOCK_ADS, true)
        set(value) = store.edit().putBoolean(KEY_BLOCK_ADS, value).apply()

    var blockWhenScreenOff: Boolean
        get() = store.getBoolean(KEY_SCREEN_OFF, true)
        set(value) = store.edit().putBoolean(KEY_SCREEN_OFF, value).apply()

    /** Sticky user intent, so a reboot can bring the tunnel back up. */
    var enabled: Boolean
        get() = store.getBoolean(KEY_ENABLED, false)
        set(value) = store.edit().putBoolean(KEY_ENABLED, value).apply()

    var shapedApps: Set<String>
        get() = store.getStringSet(KEY_APPS, null) ?: emptySet()
        set(value) = store.edit().putStringSet(KEY_APPS, value).apply()

    /**
     * On first run, preselect whichever TikTok build is actually installed.
     * Returns the resulting selection.
     */
    fun ensureDefaultApps(packageManager: PackageManager): Set<String> {
        if (store.contains(KEY_APPS)) return shapedApps
        val installed = TIKTOK_PACKAGES.filter { isInstalled(packageManager, it) }.toSet()
        shapedApps = installed
        return installed
    }

    private fun isInstalled(pm: PackageManager, packageName: String): Boolean = try {
        pm.getPackageInfo(packageName, 0)
        true
    } catch (_: PackageManager.NameNotFoundException) {
        false
    }

    companion object {
        const val DEFAULT_DOWN_KBPS = 600
        const val DEFAULT_UP_KBPS = 500

        /** Presets exposed as chips, in kbps. */
        const val PRESET_EXTREME = 250
        const val PRESET_SAVER = 600
        const val PRESET_BALANCED = 1200
        const val PRESET_LIGHT = 2500

        val TIKTOK_PACKAGES = listOf(
            "com.zhiliaoapp.musically",
            "com.zhiliaoapp.musically.go",
            "com.ss.android.ugc.trill",
            "com.ss.android.ugc.aweme",
            "com.ss.android.ugc.aweme.lite",
            "com.ss.android.ugc.tiktok.lite",
        )

        private const val KEY_DOWN = "down_kbps"
        private const val KEY_UP = "up_kbps"
        private const val KEY_CELLULAR_ONLY = "cellular_only"
        private const val KEY_BLOCK_QUIC = "block_quic"
        private const val KEY_BLOCK_ADS = "block_ads"
        private const val KEY_SCREEN_OFF = "screen_off"
        private const val KEY_ENABLED = "enabled"
        private const val KEY_APPS = "apps"
    }
}
