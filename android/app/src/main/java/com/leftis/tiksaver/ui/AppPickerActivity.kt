package com.leftis.tiksaver.ui

import android.content.Intent
import android.graphics.drawable.Drawable
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.view.View
import android.view.ViewGroup
import android.widget.BaseAdapter
import androidx.appcompat.app.AppCompatActivity
import com.leftis.tiksaver.Prefs
import com.leftis.tiksaver.databinding.ActivityAppPickerBinding
import com.leftis.tiksaver.databinding.RowAppBinding
import com.leftis.tiksaver.vpn.SaverVpnService
import java.text.Collator
import java.util.Locale
import kotlin.concurrent.thread

/**
 * Picks which apps get routed through the shaper. Only launchable apps are
 * listed, which is what the manifest `<queries>` block grants us without the
 * QUERY_ALL_PACKAGES permission.
 */
class AppPickerActivity : AppCompatActivity() {

    private class AppEntry(
        val packageName: String,
        val label: String,
        val icon: Drawable?,
    )

    private lateinit var binding: ActivityAppPickerBinding
    private lateinit var prefs: Prefs

    private val selected = mutableSetOf<String>()
    private var initialSelection: Set<String> = emptySet()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityAppPickerBinding.inflate(layoutInflater)
        setContentView(binding.root)

        prefs = Prefs(this)
        initialSelection = prefs.shapedApps
        selected += initialSelection

        binding.toolbar.setNavigationOnClickListener { finish() }
        loadApps()
    }

    override fun finish() {
        val changed = selected != initialSelection
        prefs.shapedApps = selected.toSet()
        // The allowed-application list is baked into the tun at establish()
        // time, so a change only takes effect after a rebuild.
        if (changed && SaverVpnService.isRunning) SaverVpnService.rebuild(this)
        super.finish()
    }

    private fun loadApps() {
        binding.progress.visibility = View.VISIBLE
        thread(name = "tiksaver-applist") {
            val entries = collectApps()
            Handler(Looper.getMainLooper()).post {
                if (isFinishing || isDestroyed) return@post
                binding.progress.visibility = View.GONE
                val adapter = AppAdapter(entries)
                binding.appList.adapter = adapter
                binding.appList.setOnItemClickListener { _, _, position, _ ->
                    val entry = entries[position]
                    if (!selected.remove(entry.packageName)) selected += entry.packageName
                    adapter.notifyDataSetChanged()
                }
            }
        }
    }

    private fun collectApps(): List<AppEntry> {
        val pm = packageManager
        val launcherIntent = Intent(Intent.ACTION_MAIN).addCategory(Intent.CATEGORY_LAUNCHER)

        @Suppress("DEPRECATION")
        val resolved = try {
            pm.queryIntentActivities(launcherIntent, 0)
        } catch (e: Exception) {
            emptyList()
        }

        val byPackage = LinkedHashMap<String, AppEntry>()
        for (info in resolved) {
            val packageName = info.activityInfo?.packageName ?: continue
            if (packageName == this.packageName) continue
            if (byPackage.containsKey(packageName)) continue
            val label = runCatching { info.loadLabel(pm).toString() }.getOrDefault(packageName)
            val icon = runCatching { info.loadIcon(pm) }.getOrNull()
            byPackage[packageName] = AppEntry(packageName, label, icon)
        }

        // Anything already selected but no longer launchable still deserves a row.
        for (packageName in selected) {
            if (byPackage.containsKey(packageName)) continue
            byPackage[packageName] = AppEntry(packageName, packageName, null)
        }

        val collator = Collator.getInstance(Locale.getDefault())
        return byPackage.values.sortedWith(
            compareByDescending<AppEntry> { it.packageName in selected }
                .thenByDescending { it.packageName in Prefs.TIKTOK_PACKAGES }
                .thenBy(collator) { it.label }
        )
    }

    private inner class AppAdapter(private val items: List<AppEntry>) : BaseAdapter() {

        override fun getCount(): Int = items.size

        override fun getItem(position: Int): Any = items[position]

        override fun getItemId(position: Int): Long = position.toLong()

        override fun getView(position: Int, convertView: View?, parent: ViewGroup): View {
            val row = if (convertView == null) {
                RowAppBinding.inflate(layoutInflater, parent, false)
            } else {
                RowAppBinding.bind(convertView)
            }
            val entry = items[position]
            row.appLabel.text = entry.label
            row.appPackage.text = entry.packageName
            row.appIcon.setImageDrawable(entry.icon)
            row.appCheck.isChecked = entry.packageName in selected
            return row.root
        }
    }
}
