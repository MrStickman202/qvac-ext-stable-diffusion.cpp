import java.util.Properties

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("org.jetbrains.kotlin.plugin.compose")
}

// The QAIRT SDK: its headers build the QNN sidecars, and its Android libraries go into the APK.
// Set qnn.sdk.dir in local.properties, or QNN_SDK_ROOT in the environment.
val localProps = Properties().apply {
    val f = rootProject.file("local.properties")
    if (f.exists()) f.inputStream().use { load(it) }
}
val qnnSdk: String = (localProps.getProperty("qnn.sdk.dir") ?: System.getenv("QNN_SDK_ROOT"))
    ?: throw GradleException("set qnn.sdk.dir in local.properties or QNN_SDK_ROOT to the QAIRT SDK root")

// Hexagon versions to support: v73 (Snapdragon 8 Gen 2), v75 (8 Gen 3), v79 (8 Elite), v81 (8 Elite Gen 5).
val htpArchs = listOf("73", "75", "79", "81")
val qnnLibsDir = layout.buildDirectory.dir("qnnLibs")

val copyQnnLibs by tasks.registering(Copy::class) {
    from("$qnnSdk/lib/aarch64-android") {
        include("libQnnHtp.so", "libQnnSystem.so")
        htpArchs.forEach { include("libQnnHtpV${it}Stub.so") }
    }
    htpArchs.forEach { arch ->
        from("$qnnSdk/lib/hexagon-v$arch/unsigned") { include("libQnnHtpV${arch}Skel.so") }
    }
    into(qnnLibsDir.map { it.dir("arm64-v8a") })
}

android {
    namespace = "io.tether.sdnpu"
    compileSdk = 35
    ndkVersion = "27.2.12479018"

    defaultConfig {
        applicationId = "io.tether.sdnpu"
        minSdk = 31  // Build.SOC_MODEL
        targetSdk = 35
        versionCode = 1
        versionName = "0.1"
        ndk { abiFilters += "arm64-v8a" }
        externalNativeBuild {
            cmake {
                arguments += listOf("-DQNN_SDK_ROOT=$qnnSdk", "-DCMAKE_BUILD_TYPE=Release", "-DANDROID_STL=c++_shared")
                // -PnativeJobs=N caps parallel C++ compiles (src/vocab/vocab.cpp alone needs a few GB of RAM).
                providers.gradleProperty("nativeJobs").orNull?.let {
                    arguments += listOf("-DCMAKE_JOB_POOLS=compile=$it", "-DCMAKE_JOB_POOL_COMPILE=compile")
                }
                targets += "sdjni"
            }
        }
    }

    signingConfigs {
        // A fixed, non-secret development key (password "android"), so builds from any machine can update an
        // installed copy. Use your own key for anything you publish.
        create("dev") {
            storeFile = file("../sdnpu-dev.keystore")
            storePassword = "android"
            keyAlias = "sdnpu"
            keyPassword = "android"
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            signingConfig = signingConfigs.getByName("dev")
        }
        debug {
            signingConfig = signingConfigs.getByName("dev")
            // The native code is built optimized either way: an unoptimized UNet on the CPU is unusably slow.
            externalNativeBuild { cmake { arguments += "-DCMAKE_BUILD_TYPE=Release" } }
        }
    }

    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }

    sourceSets["main"].jniLibs.srcDir(qnnLibsDir)

    packaging {
        // Extract the native libraries to nativeLibraryDir: the NPU loads libQnnHtpV*Skel.so from a real file path.
        jniLibs.useLegacyPackaging = true
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }
    buildFeatures { compose = true }
}

tasks.named("preBuild") { dependsOn(copyQnnLibs) }

dependencies {
    val composeBom = platform("androidx.compose:compose-bom:2024.06.00")
    implementation(composeBom)
    implementation("androidx.activity:activity-compose:1.9.0")
    implementation("androidx.compose.material3:material3")
    implementation("androidx.compose.material:material-icons-extended")
    implementation("androidx.compose.ui:ui")
    implementation("androidx.lifecycle:lifecycle-viewmodel-compose:2.8.2")
    implementation("androidx.lifecycle:lifecycle-runtime-compose:2.8.2")
}
