package com.leftis.tiksaver.ui

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.net.VpnService
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import com.leftis.tiksaver.Prefs
import com.leftis.tiksaver.R
import com.leftis.tiksaver.databinding.ActivityMainBinding
import com.leftis.tiksaver.formatBytes
import com.leftis.tiksaver.formatRate
import com.leftis.tiksaver.megabytesPerMinute
import com.leftis.tiksaver.net.Stats
import com.leftis.tiksaver.vpn.SaverVpnService

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private lateinit var prefs: Prefs

    private val handler = Handler(Looper.getMainLooper())

    /** Guards the chip <-> slider round trip from feeding itself. */
    private var updatingUi = false

    private val refreshTick = object : Runnable {
        override fun run() {
            refreshStatus()
            handler.postDelayed(this, REFRESH_MS)
        }
    }

    private val vpnConsent = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { result ->
        if (result.resultCode == RESULT_OK) {
            SaverVpnService.start(this)
            handler.postDelayed({ refreshStatus() }, 400)
        } else {
            toast(getString(R.string.vpn_denied))
        }
    }

    private val notificationConsent = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { /* Shaping still works without it; only the live counter goes missing. */ }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        prefs = Prefs(this)
        prefs.ensureDefaultApps(packageManager)

        setUpSwitchRows()
        setUpSpeedControls()
        setUpStatRows()

        binding.toggleButton.setOnClickListener { onToggle() }
        binding.appsCard.setOnClickListener {
            startActivity(Intent(this, AppPickerActivity::class.java))
        }
        binding.resetButton.setOnClickListener {
            Stats.reset()
            refreshStatus()
        }

        requestNotificationPermission()
    }

    override fun onResume() {
        super.onResume()
        loadPrefsIntoUi()
        handler.post(refreshTick)
    }

    override fun onPause() {
        handler.removeCallbacks(refreshTick)
        super.onPause()
    }

    // -- wiring --------------------------------------------------------------

    private fun setUpSwitchRows() {
        bindSwitch(
            row = binding.optCellular,
            title = R.string.opt_cellular_only,
            subtitle = R.string.opt_cellular_only_sub,
            read = { prefs.cellularOnly },
            write = { prefs.cellularOnly = it },
        )
        bindSwitch(
            row = binding.optQuic,
            title = R.string.opt_block_quic,
            subtitle = R.string.opt_block_quic_sub,
            read = { prefs.blockQuic },
            write = { prefs.blockQuic = it },
        )
        bindSwitch(
            row = binding.optAds,
            title = R.string.opt_block_ads,
            subtitle = R.string.opt_block_ads_sub,
            read = { prefs.blockAds },
            write = { prefs.blockAds = it },
        )
        bindSwitch(
            row = binding.optScreenOff,
            title = R.string.opt_screen_off,
            subtitle = R.string.opt_screen_off_sub,
            read = { prefs.blockWhenScreenOff },
            write = { prefs.blockWhenScreenOff = it },
        )
    }

    private fun bindSwitch(
        row: com.leftis.tiksaver.databinding.RowSwitchBinding,
        title: Int,
        subtitle: Int,
        read: () -> Boolean,
        write: (Boolean) -> Unit,
    ) {
        row.rowTitle.setText(title)
        row.rowSub.setText(subtitle)
        row.rowSwitch.isChecked = read()
        row.root.setOnClickListener {
            val next = !row.rowSwitch.isChecked
            row.rowSwitch.isChecked = next
            write(next)
            pushConfigIfRunning()
        }
    }

    private fun setUpSpeedControls() {
        binding.downSlider.addOnChangeListener { _, value, fromUser ->
            if (!fromUser || updatingUi) return@addOnChangeListener
            prefs.downKbps = value.toInt()
            updateBudgetLabels()
            syncPresetChips()
            pushConfigIfRunning()
        }
        binding.upSlider.addOnChangeListener { _, value, fromUser ->
            if (!fromUser || updatingUi) return@addOnChangeListener
            prefs.upKbps = value.toInt()
            updateBudgetLabels()
            pushConfigIfRunning()
        }

        val presets = mapOf(
            binding.presetExtreme.id to Prefs.PRESET_EXTREME,
            binding.presetSaver.id to Prefs.PRESET_SAVER,
            binding.presetBalanced.id to Prefs.PRESET_BALANCED,
            binding.presetLight.id to Prefs.PRESET_LIGHT,
        )
        binding.presetGroup.setOnCheckedStateChangeListener { _, checkedIds ->
            if (updatingUi) return@setOnCheckedStateChangeListener
            val checkedId = checkedIds.firstOrNull() ?: return@setOnCheckedStateChangeListener
            val kbps = presets[checkedId] ?: return@setOnCheckedStateChangeListener
            prefs.downKbps = kbps
            updatingUi = true
            binding.downSlider.value = kbps.toFloat()
            updatingUi = false
            updateBudgetLabels()
            pushConfigIfRunning()
        }
    }

    private fun setUpStatRows() {
        binding.statDown.statLabel.setText(R.string.stats_down)
        binding.statUp.statLabel.setText(R.string.stats_up)
        binding.statRate.statLabel.setText(R.string.stats_rate)
        binding.statBlocked.statLabel.setText(R.string.stats_blocked)
        binding.statConns.statLabel.setText(R.string.stats_conns)
    }

    private fun loadPrefsIntoUi() {
        updatingUi = true
        binding.downSlider.value = clampToSlider(binding.downSlider, prefs.downKbps)
        binding.upSlider.value = clampToSlider(binding.upSlider, prefs.upKbps)
        binding.optCellular.rowSwitch.isChecked = prefs.cellularOnly
        binding.optQuic.rowSwitch.isChecked = prefs.blockQuic
        binding.optAds.rowSwitch.isChecked = prefs.blockAds
        binding.optScreenOff.rowSwitch.isChecked = prefs.blockWhenScreenOff
        updatingUi = false

        syncPresetChips()
        updateBudgetLabels()
        refreshStatus()
    }

    private fun clampToSlider(slider: com.google.android.material.slider.Slider, kbps: Int): Float =
        kbps.toFloat().coerceIn(slider.valueFrom, slider.valueTo)

    private fun syncPresetChips() {
        updatingUi = true
        val id = when (prefs.downKbps) {
            Prefs.PRESET_EXTREME -> binding.presetExtreme.id
            Prefs.PRESET_SAVER -> binding.presetSaver.id
            Prefs.PRESET_BALANCED -> binding.presetBalanced.id
            Prefs.PRESET_LIGHT -> binding.presetLight.id
            else -> null
        }
        if (id != null) binding.presetGroup.check(id) else binding.presetGroup.clearCheck()
        updatingUi = false
    }

    private fun updateBudgetLabels() {
        binding.budgetText.text =
            getString(R.string.budget_fmt, prefs.downKbps, megabytesPerMinute(prefs.downKbps))
        binding.upBudgetText.text =
            getString(R.string.budget_fmt, prefs.upKbps, megabytesPerMinute(prefs.upKbps))
    }

    // -- state ---------------------------------------------------------------

    private fun onToggle() {
        if (SaverVpnService.isRunning) {
            SaverVpnService.stop(this)
            handler.postDelayed({ refreshStatus() }, 400)
            return
        }

        if (prefs.shapedApps.isEmpty()) {
            toast(getString(R.string.apps_none))
            startActivity(Intent(this, AppPickerActivity::class.java))
            return
        }

        val consent = try {
            VpnService.prepare(this)
        } catch (e: Exception) {
            toast(getString(R.string.vpn_unsupported))
            return
        }

        if (consent == null) {
            SaverVpnService.start(this)
            handler.postDelayed({ refreshStatus() }, 400)
        } else {
            vpnConsent.launch(consent)
        }
    }

    /** Re-sends the current settings so a running tunnel picks them up live. */
    private fun pushConfigIfRunning() {
        if (SaverVpnService.isRunning) SaverVpnService.start(this)
    }

    private fun refreshStatus() {
        val running = SaverVpnService.isRunning
        val paused = running && SaverVpnService.isPausedOnWifi

        binding.statusText.setText(
            when {
                paused -> R.string.status_paused_wifi
                running -> R.string.status_on
                else -> R.string.status_off
            }
        )
        binding.statusSub.text =
            getString(R.string.budget_fmt, prefs.downKbps, megabytesPerMinute(prefs.downKbps))
        binding.toggleButton.setText(if (running) R.string.btn_stop else R.string.btn_start)

        val apps = prefs.shapedApps
        binding.appsSub.text = when {
            apps.isEmpty() && Prefs.TIKTOK_PACKAGES.none { isInstalled(it) } -> getString(R.string.apps_missing)
            apps.isEmpty() -> getString(R.string.apps_none)
            else -> getString(R.string.apps_sub, apps.size)
        }

        Stats.decayIfIdle()
        binding.statDown.statValue.text = formatBytes(Stats.bytesDown.get())
        binding.statUp.statValue.text = formatBytes(Stats.bytesUp.get())
        binding.statRate.statValue.text = formatRate(Stats.lastRateBytesPerSecond)
        binding.statBlocked.statValue.text = Stats.blocked.get().toString()
        binding.statConns.statValue.text = Stats.openConnections.get().toString()
    }

    private fun isInstalled(packageName: String): Boolean = try {
        packageManager.getPackageInfo(packageName, 0)
        true
    } catch (_: PackageManager.NameNotFoundException) {
        false
    }

    private fun requestNotificationPermission() {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.TIRAMISU) return
        val granted = ContextCompat.checkSelfPermission(this, Manifest.permission.POST_NOTIFICATIONS) ==
            PackageManager.PERMISSION_GRANTED
        if (!granted) notificationConsent.launch(Manifest.permission.POST_NOTIFICATIONS)
    }

    private fun toast(message: String) =
        Toast.makeText(this, message, Toast.LENGTH_LONG).show()

    private companion object {
        const val REFRESH_MS = 1000L
    }
}
