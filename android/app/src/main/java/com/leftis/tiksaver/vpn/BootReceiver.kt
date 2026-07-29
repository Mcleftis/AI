package com.leftis.tiksaver.vpn

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.net.VpnService
import android.util.Log
import com.leftis.tiksaver.Prefs

/**
 * Brings shaping back after a reboot or an app update, but only when the user
 * had it on and VPN consent is still granted — we can never prompt from here.
 */
class BootReceiver : BroadcastReceiver() {

    override fun onReceive(context: Context, intent: Intent) {
        when (intent.action) {
            Intent.ACTION_BOOT_COMPLETED, Intent.ACTION_MY_PACKAGE_REPLACED -> Unit
            else -> return
        }

        val prefs = Prefs(context)
        if (!prefs.enabled || prefs.shapedApps.isEmpty()) return

        if (VpnService.prepare(context) != null) {
            Log.i(TAG, "VPN consent no longer granted; not restarting")
            return
        }

        try {
            SaverVpnService.start(context)
        } catch (e: Exception) {
            Log.w(TAG, "could not restart shaping: ${e.message}")
        }
    }

    private companion object {
        const val TAG = "TikSaver/boot"
    }
}
