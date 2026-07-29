# Minification is disabled for this build type; these rules exist so that
# turning it on later does not silently break the service entry points.
-keep class com.leftis.tiksaver.vpn.SaverVpnService { *; }
-keepclassmembers class * extends android.content.BroadcastReceiver { *; }
