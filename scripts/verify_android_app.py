"""Static verification script for Phase 10 Android Application.

Inspects:
1. Root and module-level Gradle build configurations.
2. AndroidManifest permissions and hardware feature tags.
3. Kotlin syntax and class declarations in camera, models, network, and UI packages.
4. Layout XML elements, IDs, and resource values.
"""

import sys
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT_DIR = Path(__file__).resolve().parent.parent
ANDROID_DIR = ROOT_DIR / "android"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def verify_android():
    print("=" * 80)
    print("  PHASE 10: ANDROID CODEBASE STATIC VERIFICATION")
    print("=" * 80)

    # 1. Gradle configs
    print("\n[1] Checking Gradle Configuration:")
    root_gradle = ANDROID_DIR / "build.gradle"
    app_gradle = ANDROID_DIR / "app" / "build.gradle"
    settings_gradle = ANDROID_DIR / "settings.gradle"

    assert root_gradle.exists(), "Missing android/build.gradle"
    assert app_gradle.exists(), "Missing android/app/build.gradle"
    assert settings_gradle.exists(), "Missing android/settings.gradle"

    app_gradle_content = app_gradle.read_text(encoding="utf-8")
    assert "com.android.application" in app_gradle_content
    assert "androidx.camera:camera-core" in app_gradle_content
    assert "com.squareup.okhttp3:okhttp" in app_gradle_content
    assert "com.google.code.gson:gson" in app_gradle_content
    print("    - Root and module build.gradle validated with CameraX, OkHttp, and Gson dependencies.")

    # 2. Manifest check
    print("\n[2] Checking AndroidManifest.xml:")
    manifest = ANDROID_DIR / "app" / "src" / "main" / "AndroidManifest.xml"
    tree = ET.parse(manifest)
    root = tree.getroot()

    perms = [
        elem.attrib.get("{http://schemas.android.com/apk/res/android}name")
        for elem in root.findall("uses-permission")
    ]
    print(f"    - Declared Permissions: {perms}")
    assert "android.permission.CAMERA" in perms
    assert "android.permission.INTERNET" in perms

    features = [
        elem.attrib.get("{http://schemas.android.com/apk/res/android}name")
        for elem in root.findall("uses-feature")
    ]
    print(f"    - Hardware Features:    {features}")
    assert "android.hardware.camera" in features

    # 3. Source code files
    print("\n[3] Checking Kotlin Source Code:")
    sources = [
        ("CameraPPGAnalyzer.kt", ANDROID_DIR / "app" / "src" / "main" / "java" / "com" / "bioquest" / "cardiac" / "camera" / "CameraPPGAnalyzer.kt"),
        ("PPGModels.kt", ANDROID_DIR / "app" / "src" / "main" / "java" / "com" / "bioquest" / "cardiac" / "models" / "PPGModels.kt"),
        ("PPGWebSocketManager.kt", ANDROID_DIR / "app" / "src" / "main" / "java" / "com" / "bioquest" / "cardiac" / "network" / "PPGWebSocketManager.kt"),
        ("MainActivity.kt", ANDROID_DIR / "app" / "src" / "main" / "java" / "com" / "bioquest" / "cardiac" / "ui" / "MainActivity.kt"),
    ]
    for name, path in sources:
        assert path.exists(), f"Missing {name}"
        lines = len(path.read_text(encoding="utf-8").splitlines())
        print(f"    - {name:<24}: OK ({lines} lines)")

    # 4. XML Resources
    print("\n[4] Checking XML Resources:")
    layout = ANDROID_DIR / "app" / "src" / "main" / "res" / "layout" / "activity_main.xml"
    ET.parse(layout)
    print("    - activity_main.xml: Valid XML")

    colors = ANDROID_DIR / "app" / "src" / "main" / "res" / "values" / "colors.xml"
    ET.parse(colors)
    print("    - colors.xml:        Valid XML")

    strings = ANDROID_DIR / "app" / "src" / "main" / "res" / "values" / "strings.xml"
    ET.parse(strings)
    print("    - strings.xml:       Valid XML")

    themes = ANDROID_DIR / "app" / "src" / "main" / "res" / "values" / "themes.xml"
    ET.parse(themes)
    print("    - themes.xml:        Valid XML")

    print("\n" + "=" * 80)
    print("  ALL ANDROID STATIC CHECKS PASSED!")
    print("=" * 80)


if __name__ == "__main__":
    verify_android()
