# TikSaver

An Android app that cuts how much **mobile data** TikTok burns, without root.

## What it actually does

TikTok picks its video quality from how fast the network feels. On a good LTE/5G
link it happily pulls 1080p at several megabits, and it preloads the next clips
while you scroll — so a lot of what you pay for is video you never watch.

TikSaver puts a local, per-app bandwidth shaper in front of it:

| Lever | Effect |
|---|---|
| **Speed cap** | A token bucket paces TikTok's downloads. Its adaptive bitrate logic then settles on a low-bitrate rendition instead of the 1080p one. This is where most of the saving comes from. |
| **Force TCP (block QUIC)** | TikTok's QUIC traffic cannot be shaped through TCP flow control, so UDP/443 is refused with an ICMP port-unreachable and the app falls back to TCP immediately. |
| **Ad / telemetry blocking** | Known ad, attribution and log endpoints are dropped, matched both on DNS queries and on the TLS SNI (so it still works when TikTok resolves names through its own HTTPDNS). |
| **Screen-off cut** | While the screen is off, TikTok's flows are reset so background prefetch stops. |
| **Mobile-data only** | On Wi-Fi the tunnel is torn down completely and traffic runs at full speed. |

The cap is the honest number: at 600 kbps TikTok cannot exceed **4.5 MB per
minute**, no matter what it tries to fetch. The app shows that budget next to
the slider.

## How it works

`VpnService` gives us a tun device with `addAllowedApplication()`, so **only the
apps you select** are routed through it — everything else on the phone is
untouched, and a bug here can only ever break TikTok.

Packets are terminated in userspace (`app/src/main/java/com/leftis/tiksaver/net/`):

- `NatEngine` — three threads (tun reader, NIO selector worker, tun writer). All
  session state lives on the worker thread, so nothing needs locking.
- `TcpSession` — a small TCP implementation. Terminating the connection instead
  of forwarding it is what creates the throttle point: the rate at which we
  drain the remote socket becomes the rate the app sees, and TCP's own flow
  control pushes the slowdown back to the server without dropping a packet.
- `UdpSession` — UDP has no flow control, so shaping there means dropping, which
  is exactly the signal a congestion-controlled protocol expects.
- `TokenBucket` — byte-granular pacing with a quarter-second burst, small enough
  that a video chunk cannot slip through but large enough to keep API calls
  responsive.

IPv6 is routed into the tunnel and dropped on purpose: without that route it
would escape the shaper over a second address family.

## Building

```
cd android
./gradlew assembleRelease
```

CI (`.github/workflows/android-apk.yml`) builds the same APK and attaches it to
the `tiksaver-latest` release.

### About the signing key

`keystore/tiksaver-sideload.jks` (password `tiksaver`) is committed on purpose.
It exists so that a new sideloaded build installs over the previous one instead
of forcing an uninstall. It is a throwaway: it protects nothing, and it must
never be used to sign anything published to an app store.

## Installing

1. Download `TikSaver.apk` from the release.
2. Allow installs from the browser/file manager when Android asks.
3. Open the app, press **Start**, and accept the VPN prompt. Android will show
   the usual key icon — this VPN is local; nothing is sent to any server.

## Limits worth knowing

- Video quality drops. That is the mechanism, not a side effect.
- If TikTok ever fails to load, the first things to try are raising the cap and
  turning off ad blocking.
- The shaper cannot see inside TLS, so it cannot tell a preloaded video from one
  you are watching — it limits the total instead.
