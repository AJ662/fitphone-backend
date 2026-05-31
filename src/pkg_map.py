"""Maps Android package names to the human-readable app names the LSApp
model was trained on (87-item Android vocabulary: 'Instagram', 'Reddit',
'Settings', etc.). The /predict endpoint normalizes incoming history
through this so the phone can send raw package names."""

PKG_TO_NAME: dict[str, str] = {
    # social
    "com.instagram.android": "Instagram",
    "com.facebook.katana": "Facebook",
    "com.snapchat.android": "Snapchat",
    "com.twitter.android": "Twitter",
    "com.reddit.frontpage": "Reddit",
    "com.pinterest": "Pinterest",
    "com.quora.android": "Quora",
    # messaging
    "com.whatsapp": "WhatsApp Messenger",
    "com.facebook.orca": "Facebook Messenger",
    "com.facebook.mlite": "Messenger Lite",
    "com.google.android.apps.messaging": "Messages",
    "com.android.messaging": "Messaging",
    "org.telegram.messenger": "Telegram",
    "com.discord": "Discord",
    "com.google.android.talk": "Hangouts",
    "kik.android": "Kik",
    "com.tencent.mm": "WeChat",
    # video
    "com.google.android.youtube": "YouTube",
    "com.netflix.mediaclient": "Netflix",
    "com.hulu.plus": "Hulu",
    # shopping
    "com.amazon.mShop.android.shopping": "Amazon Shopping",
    "com.walmart.android": "Walmart",
    "com.ebay.mobile": "eBay",
    "com.offerup": "OfferUp",
    "com.paypal.android.p2pmobile": "PayPal Mobile Cash",
    "com.ibotta.android": "Ibotta",
    # games (samples only)
    "com.zynga.wwf2.free": "Words With Friends 2",
    # browsers / utilities (in LSApp vocab)
    "com.android.chrome": "Google Chrome",
    "com.brave.browser": "Brave Browser",
    "com.sec.android.app.sbrowser": "Samsung Internet Browser",
    "com.android.calculator2": "Calculator",
    "com.google.android.calculator": "Calculator",
    "com.android.deskclock": "Clock",
    "com.google.android.calendar": "Calendar",
    "com.android.contacts": "Contacts",
    "com.android.camera": "Camera",
    "com.android.camera2": "Camera",
    "com.google.android.GoogleCamera": "Camera",
    "com.google.android.dialer": "Phone",
    "com.android.dialer": "Phone",
    "com.google.android.apps.maps": "Maps",
    "com.google.android.gm": "Gmail",
    "com.microsoft.office.outlook": "Microsoft Outlook",
    "com.google.android.googlequicksearchbox": "Google",
    "com.google.android.apps.photos": "Google Photos",
    "com.google.android.music": "Google Play Music",
    "com.android.vending": "Google Play Store",
    "com.spotify.music": "Spotify Music",
    "com.android.settings": "Settings",
    "com.samsung.android.email.provider": "Samsung Email",
    "com.sec.android.gallery3d": "Samsung Gallery",
    "com.samsung.android.app.notes": "Samsung Notes",
    "com.samsung.android.spay": "Samsung Pay",
}


def normalize(name: str) -> str:
    """Return the LSApp app name for a package, or the input unchanged."""
    return PKG_TO_NAME.get(name, name)


NOISE_PACKAGES: set[str] = {
    "com.google.android.apps.nexuslauncher",
    "com.sec.android.app.launcher",
    "com.android.launcher",
    "com.android.launcher3",
    "com.android.systemui",
    "com.example.fitphone2",
    "com.tailscale.ipn",
}
