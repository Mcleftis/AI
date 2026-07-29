plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
}

base {
    archivesName.set("TikSaver")
}

android {
    namespace = "com.leftis.tiksaver"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.leftis.tiksaver"
        minSdk = 24
        targetSdk = 35
        versionCode = 1
        versionName = "1.0"
        resourceConfigurations += setOf("en", "el")
    }

    signingConfigs {
        create("sideload") {
            // Throwaway key used only so that sideloaded updates install over
            // each other. It is intentionally in the repository; it protects
            // nothing and must never be used for a Play Store upload.
            storeFile = rootProject.file("keystore/tiksaver-sideload.jks")
            storePassword = "tiksaver"
            keyAlias = "tiksaver"
            keyPassword = "tiksaver"
            enableV1Signing = true
            enableV2Signing = true
            enableV3Signing = true
        }
    }

    buildTypes {
        debug {
            signingConfig = signingConfigs.getByName("sideload")
            isMinifyEnabled = false
        }
        release {
            signingConfig = signingConfigs.getByName("sideload")
            // The packet engine is reflection-free, but keeping symbols makes
            // on-device logcat triage of the TCP stack actually readable.
            isMinifyEnabled = false
            isShrinkResources = false
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
    }

    buildFeatures {
        viewBinding = true
    }

    packaging {
        resources.excludes += setOf("/META-INF/{AL2.0,LGPL2.1}")
    }
}

dependencies {
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.appcompat)
    implementation(libs.material)
    implementation(libs.androidx.activity)
    implementation(libs.androidx.constraintlayout)
    implementation(libs.androidx.lifecycle.runtime.ktx)
}
