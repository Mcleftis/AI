package com.leftis.tiksaver.vpn

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.content.pm.PackageManager
import android.content.pm.ServiceInfo
import android.net.ConnectivityManager
import android.net.Network
import android.net.NetworkCapabilities
import android.net.NetworkRequest
import android.net.VpnService
import android.os.Build
import android.os.Handler
import android.os.Looper
import android.os.ParcelFileDescriptor
import android.os.PowerManager
import android.util.Log
import androidx.core.app.NotificationCompat
import androidx.core.content.ContextCompat
import com.leftis.tiksaver.Prefs
import com.leftis.tiksaver.R
import com.leftis.tiksaver.formatBytes
import com.leftis.tiksaver.net.EngineConfig
import com.leftis.tiksaver.net.NatEngine
import com.leftis.tiksaver.net.SocketProtector
import com.leftis.tiksaver.net.Stats
import com.leftis.tiksaver.net.bytesToIp
import com.leftis.tiksaver.ui.MainActivity
import java.net.DatagramSocket
import java.net.Inet4Address
import java.net.InetAddress
import java.net.Socket

/**
 * Owns the tun device and the packet engine.
 *
 * The service stays alive for as long as the user wants shaping, but the tunnel
 * itself comes and goes: on Wi-Fi (with "mobile data only" enabled) we tear the
 * tun down entirely rather than pass traffic through an idle shaper, which is
 * both faster for the user and cheaper for the battery.
 */
class SaverVpnService : VpnService(), SocketProtector {

    private lateinit var prefs: Prefs
    private val handler = Handler(Looper.getMainLooper())

    private var engine: NatEngine? = null
    private var connectivityCallback: ConnectivityManager.NetworkCallback? = null
    private var screenReceiver: BroadcastReceiver? = null

    private val validatedWifi = mutableSetOf<Network>()

    private val notificationTicker = object : Runnable {
        override fun run() {
            if (!isRunning) return
            notificationManager().notify(NOTIFICATION_ID, buildNotification())
            handler.postDelayed(this, NOTIFICATION_INTERVAL_MS)
        }
    }

    override fun onCreate() {
        super.onCreate()
        prefs = Prefs(this)
        createNotificationChannel()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        // Every entry point reaches us through startForegroundService(), which
        // obliges us to enter the foreground promptly — including on the way to
        // shutting down.
        startForegroundCompat()

        when (intent?.action) {
            ACTION_STOP -> {
                prefs.enabled = false
                shutdown()
                stopSelf()
                return START_NOT_STICKY
            }

            ACTION_REBUILD -> if (isRunning) {
                // The allowed-application set is fixed at establish() time.
                stopTunnel()
                evaluateTunnel()
                return START_STICKY
            }
        }
        startShaping()
        return START_STICKY
    }

    override fun onDestroy() {
        shutdown()
        super.onDestroy()
    }

    /** The user revoked VPN consent, or another VPN took over. */
    override fun onRevoke() {
        prefs.enabled = false
        shutdown()
        stopSelf()
        super.onRevoke()
    }

    // -- SocketProtector -----------------------------------------------------

    override fun protect(socket: Socket): Boolean = super<VpnService>.protect(socket)

    override fun protect(socket: DatagramSocket): Boolean = super<VpnService>.protect(socket)

    // -- lifecycle -----------------------------------------------------------

    private fun startShaping() {
        if (isRunning) {
            // Already up: this is a settings change, so just re-apply them.
            applyConfig()
            evaluateTunnel()
            return
        }

        isRunning = true
        prefs.enabled = true
        Stats.reset()
        registerScreenReceiver()
        registerConnectivityCallback()
        evaluateTunnel()
        handler.postDelayed(notificationTicker, NOTIFICATION_INTERVAL_MS)
    }

    private fun shutdown() {
        handler.removeCallbacks(notificationTicker)
        unregisterConnectivityCallback()
        unregisterScreenReceiver()
        stopTunnel()
        isRunning = false
        isPausedOnWifi = false
        stopForegroundCompat()
    }

    /** Decides whether the tun should currently exist, and makes it so. */
    private fun evaluateTunnel() {
        val pause = prefs.cellularOnly && validatedWifi.isNotEmpty()
        isPausedOnWifi = pause
        if (pause) stopTunnel() else startTunnel()
        notificationManager().notify(NOTIFICATION_ID, buildNotification())
    }

    private fun startTunnel() {
        if (engine != null) return
        val tun = establishTun() ?: run {
            Log.w(TAG, "could not establish tun (no shaped app installed?)")
            return
        }
        val created = NatEngine(tun, currentConfig(), this)
        created.blackout = prefs.blockWhenScreenOff && !isScreenOn()
        engine = created
        created.start()
        Log.i(TAG, "tunnel up at ${prefs.downKbps} kbps down / ${prefs.upKbps} kbps up")
    }

    private fun stopTunnel() {
        engine?.stop()
        engine = null
    }

    private fun applyConfig() {
        engine?.let {
            it.updateConfig(currentConfig())
            it.blackout = prefs.blockWhenScreenOff && !isScreenOn()
        }
    }

    private fun establishTun(): ParcelFileDescriptor? {
        val apps = prefs.shapedApps
        if (apps.isEmpty()) return null

        val builder = Builder()
            .setSession(getString(R.string.app_name))
            .setMtu(MTU)
            .addAddress(TUN_ADDRESS, 32)
            .addRoute("0.0.0.0", 0)
            .addDnsServer(VIRTUAL_DNS)

        builder.setBlocking(true)

        // IPv6 is routed in only so that it cannot slip past the shaper; the
        // engine drops it and the app falls back to IPv4.
        try {
            builder.addAddress(TUN_ADDRESS_V6, 128)
            builder.addRoute("::", 0)
        } catch (e: IllegalArgumentException) {
            Log.w(TAG, "no IPv6 inside the tunnel: ${e.message}")
        }

        var allowed = 0
        for (packageName in apps) {
            try {
                builder.addAllowedApplication(packageName)
                allowed++
            } catch (_: PackageManager.NameNotFoundException) {
                Log.w(TAG, "shaped app not installed: $packageName")
            }
        }
        if (allowed == 0) return null

        builder.setConfigureIntent(
            PendingIntent.getActivity(
                this,
                0,
                Intent(this, MainActivity::class.java),
                PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
            )
        )

        return try {
            val descriptor = builder.establish()
            if (descriptor != null) {
                // null == "follow the system default", which keeps data
                // accounting attributed to the real network.
                setUnderlyingNetworks(null)
            }
            descriptor
        } catch (e: Exception) {
            Log.e(TAG, "establish() failed", e)
            null
        }
    }

    private fun currentConfig(): EngineConfig = EngineConfig(
        downKbps = prefs.downKbps,
        upKbps = prefs.upKbps,
        blockQuic = prefs.blockQuic,
        blockAds = prefs.blockAds,
        upstreamDns = systemDnsServers(),
        virtualDns = bytesToIp(InetAddress.getByName(VIRTUAL_DNS).address),
    )

    /**
     * DNS servers of the real (non-VPN) network, so intercepted queries go
     * somewhere the carrier actually routes.
     */
    @Suppress("DEPRECATION")
    private fun systemDnsServers(): List<InetAddress> {
        val cm = getSystemService(ConnectivityManager::class.java) ?: return FALLBACK_DNS
        val servers = mutableListOf<InetAddress>()
        try {
            for (network in cm.allNetworks) {
                val caps = cm.getNetworkCapabilities(network) ?: continue
                if (!caps.hasCapability(NetworkCapabilities.NET_CAPABILITY_INTERNET)) continue
                if (!caps.hasCapability(NetworkCapabilities.NET_CAPABILITY_NOT_VPN)) continue
                val properties = cm.getLinkProperties(network) ?: continue
                servers += properties.dnsServers.filterIsInstance<Inet4Address>()
            }
        } catch (e: Exception) {
            Log.w(TAG, "could not read system DNS: ${e.message}")
        }
        return servers.ifEmpty { FALLBACK_DNS }
    }

    // -- connectivity and screen state ---------------------------------------

    private fun registerConnectivityCallback() {
        val cm = getSystemService(ConnectivityManager::class.java) ?: return
        val request = NetworkRequest.Builder()
            .addCapability(NetworkCapabilities.NET_CAPABILITY_INTERNET)
            .addCapability(NetworkCapabilities.NET_CAPABILITY_NOT_VPN)
            .build()

        val callback = object : ConnectivityManager.NetworkCallback() {
            override fun onCapabilitiesChanged(network: Network, caps: NetworkCapabilities) {
                val usableWifi = caps.hasTransport(NetworkCapabilities.TRANSPORT_WIFI) &&
                    caps.hasCapability(NetworkCapabilities.NET_CAPABILITY_VALIDATED)
                val changed = if (usableWifi) validatedWifi.add(network) else validatedWifi.remove(network)
                if (changed) postNetworkChange()
            }

            override fun onLost(network: Network) {
                if (validatedWifi.remove(network)) postNetworkChange()
                else postNetworkChange() // the underlying cellular link may have moved
            }
        }

        try {
            cm.registerNetworkCallback(request, callback)
            connectivityCallback = callback
        } catch (e: Exception) {
            Log.w(TAG, "could not watch connectivity: ${e.message}")
        }
    }

    private fun postNetworkChange() = handler.post {
        if (!isRunning) return@post
        evaluateTunnel()
        applyConfig()
    }

    private fun unregisterConnectivityCallback() {
        val cm = getSystemService(ConnectivityManager::class.java)
        connectivityCallback?.let {
            try {
                cm?.unregisterNetworkCallback(it)
            } catch (_: Exception) {
            }
        }
        connectivityCallback = null
        validatedWifi.clear()
    }

    private fun registerScreenReceiver() {
        val receiver = object : BroadcastReceiver() {
            override fun onReceive(context: Context?, intent: Intent?) {
                val screenOn = intent?.action == Intent.ACTION_SCREEN_ON
                engine?.blackout = prefs.blockWhenScreenOff && !screenOn
            }
        }
        val filter = IntentFilter().apply {
            addAction(Intent.ACTION_SCREEN_ON)
            addAction(Intent.ACTION_SCREEN_OFF)
        }
        ContextCompat.registerReceiver(this, receiver, filter, ContextCompat.RECEIVER_NOT_EXPORTED)
        screenReceiver = receiver
    }

    private fun unregisterScreenReceiver() {
        screenReceiver?.let {
            try {
                unregisterReceiver(it)
            } catch (_: IllegalArgumentException) {
            }
        }
        screenReceiver = null
    }

    private fun isScreenOn(): Boolean {
        val pm = getSystemService(PowerManager::class.java) ?: return true
        return pm.isInteractive
    }

    // -- notification --------------------------------------------------------

    private fun notificationManager(): NotificationManager =
        getSystemService(NotificationManager::class.java)

    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.O) return
        val channel = NotificationChannel(
            CHANNEL_ID,
            getString(R.string.notif_channel),
            NotificationManager.IMPORTANCE_LOW,
        ).apply {
            setShowBadge(false)
            enableVibration(false)
        }
        notificationManager().createNotificationChannel(channel)
    }

    private fun buildNotification(): Notification {
        val content = if (isPausedOnWifi) {
            getString(R.string.status_paused_wifi)
        } else {
            getString(
                R.string.notif_text,
                "${prefs.downKbps} kbps",
                formatBytes(Stats.bytesDown.get()),
                formatBytes(Stats.bytesUp.get()),
            )
        }

        val open = PendingIntent.getActivity(
            this,
            0,
            Intent(this, MainActivity::class.java),
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
        )
        val stop = PendingIntent.getService(
            this,
            1,
            Intent(this, SaverVpnService::class.java).setAction(ACTION_STOP),
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
        )

        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setSmallIcon(R.drawable.ic_notification)
            .setContentTitle(getString(R.string.notif_title))
            .setContentText(content)
            .setContentIntent(open)
            .addAction(0, getString(R.string.notif_stop), stop)
            .setOngoing(true)
            .setOnlyAlertOnce(true)
            .setPriority(NotificationCompat.PRIORITY_LOW)
            .setCategory(NotificationCompat.CATEGORY_SERVICE)
            .build()
    }

    private fun startForegroundCompat() {
        val notification = buildNotification()
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.UPSIDE_DOWN_CAKE) {
            startForeground(NOTIFICATION_ID, notification)
            return
        }
        // Android 14 wants a declared type. VPNs are a system-exempted use; if
        // the platform disagrees, fall back to the generic special-use type.
        try {
            startForeground(
                NOTIFICATION_ID,
                notification,
                ServiceInfo.FOREGROUND_SERVICE_TYPE_SYSTEM_EXEMPTED,
            )
        } catch (e: Exception) {
            Log.w(TAG, "systemExempted foreground rejected: ${e.message}")
            try {
                startForeground(
                    NOTIFICATION_ID,
                    notification,
                    ServiceInfo.FOREGROUND_SERVICE_TYPE_SPECIAL_USE,
                )
            } catch (e2: Exception) {
                Log.e(TAG, "could not enter the foreground", e2)
            }
        }
    }

    private fun stopForegroundCompat() {
        stopForeground(STOP_FOREGROUND_REMOVE)
    }

    companion object {
        private const val TAG = "TikSaver/service"

        const val ACTION_STOP = "com.leftis.tiksaver.action.STOP"
        const val ACTION_REBUILD = "com.leftis.tiksaver.action.REBUILD"

        private const val CHANNEL_ID = "shaping"
        private const val NOTIFICATION_ID = 4711
        private const val NOTIFICATION_INTERVAL_MS = 2000L

        private const val MTU = 1500
        private const val TUN_ADDRESS = "10.111.222.1"
        private const val TUN_ADDRESS_V6 = "fd00:1b2e:5a71:1::1"
        private const val VIRTUAL_DNS = "10.111.222.3"

        private val FALLBACK_DNS: List<InetAddress> by lazy {
            listOf(
                InetAddress.getByAddress(byteArrayOf(1, 1, 1, 1)),
                InetAddress.getByAddress(byteArrayOf(8, 8, 8, 8)),
            )
        }

        @Volatile
        var isRunning: Boolean = false
            private set

        @Volatile
        var isPausedOnWifi: Boolean = false
            private set

        fun start(context: Context) {
            val intent = Intent(context, SaverVpnService::class.java)
            ContextCompat.startForegroundService(context, intent)
        }

        fun stop(context: Context) {
            val intent = Intent(context, SaverVpnService::class.java).setAction(ACTION_STOP)
            ContextCompat.startForegroundService(context, intent)
        }

        /** Re-establishes the tun, which is required after the app list changes. */
        fun rebuild(context: Context) {
            val intent = Intent(context, SaverVpnService::class.java).setAction(ACTION_REBUILD)
            ContextCompat.startForegroundService(context, intent)
        }
    }
}
