#!/usr/bin/env bash
# Pushes sd-bench (or sd-cli / test-qnn), the QNN runtime libraries and model files to an Android phone over adb,
# then runs the command there with the libraries set up for the Hexagon NPU. See README.md.
#
#   examples/android-bench/run_on_device.sh [--push FILE]... -- bench sdxs-f16.gguf sdxs_unet_fp16.bin s25_npu
#
# Environment:
#   QNN_SDK_ROOT   QAIRT SDK root (the same version that compiled the .bin files)
#   BUILD_DIR      where the Android build put bin/ (default: build-android)
#   DEVICE_DIR     working directory on the phone (default: /data/local/tmp/sdbench)
#   HTP_ARCH       Hexagon arch (v73, v75, v79, v81); detected from ro.soc.model when unset
#   ADB            adb binary (default: adb)
set -euo pipefail

ADB=${ADB:-adb}
BUILD_DIR=${BUILD_DIR:-build-android}
DEVICE_DIR=${DEVICE_DIR:-/data/local/tmp/sdbench}
: "${QNN_SDK_ROOT:?set QNN_SDK_ROOT to the QAIRT SDK root}"

push_files=()
while [[ $# -gt 0 && "$1" != "--" ]]; do
    case "$1" in
        --push) push_files+=("$2"); shift 2 ;;
        *) echo "unknown option $1 (put the command after --)" >&2; exit 2 ;;
    esac
done
[[ "${1:-}" == "--" ]] && shift
if [[ $# -eq 0 ]]; then
    echo "usage: $0 [--push FILE]... -- bench|cli|test ARGS..." >&2
    exit 2
fi

soc=$($ADB shell getprop ro.soc.model | tr -d '\r')
if [[ -z "${HTP_ARCH:-}" ]]; then
    case "$soc" in
        SM8850*) HTP_ARCH=v81 ;;
        SM8750*) HTP_ARCH=v79 ;;
        SM8650*) HTP_ARCH=v75 ;;
        SM8550*) HTP_ARCH=v73 ;;
        *) echo "unknown SoC '$soc' (not a Snapdragon 8 Gen 2 or newer?); set HTP_ARCH" >&2; exit 1 ;;
    esac
fi
arch_num=${HTP_ARCH#v}
echo "device: $($ADB shell getprop ro.product.model | tr -d '\r'), SoC $soc, Hexagon $HTP_ARCH"

lib="$QNN_SDK_ROOT/lib/aarch64-android"
skel="$QNN_SDK_ROOT/lib/hexagon-$HTP_ARCH/unsigned/libQnnHtpV${arch_num}Skel.so"
$ADB shell mkdir -p "$DEVICE_DIR"
for f in "$lib/libQnnHtp.so" "$lib/libQnnSystem.so" "$lib/libQnnHtpV${arch_num}Stub.so" "$skel" \
         "$BUILD_DIR/bin/sd-bench" "$BUILD_DIR/bin/sd-cli" "$BUILD_DIR/bin/test-qnn"; do
    [[ -f "$f" ]] && $ADB push --sync "$f" "$DEVICE_DIR/" > /dev/null
done
for f in "${push_files[@]}"; do
    echo "pushing $f"
    $ADB push --sync "$f" "$DEVICE_DIR/"
done

cmd=$1
shift
case "$cmd" in
    bench) exe=./sd-bench; args=(bench "$@") ;;
    cli) exe=./sd-cli; args=("$@") ;;
    test) exe=./test-qnn; args=("$@") ;;
    *) echo "unknown command $cmd (bench, cli or test)" >&2; exit 2 ;;
esac
quoted=$(printf ' %q' "${args[@]}")
$ADB shell "cd $DEVICE_DIR && chmod +x sd-bench sd-cli test-qnn 2>/dev/null; \
    LD_LIBRARY_PATH=$DEVICE_DIR SD_QNN_LIB_DIR=$DEVICE_DIR ADSP_LIBRARY_PATH=$DEVICE_DIR \
    ${SD_QNN_PERF:+SD_QNN_PERF=$SD_QNN_PERF} ${SD_QNN_LOG:+SD_QNN_LOG=$SD_QNN_LOG} ${SDBENCH_VERBOSE:+SDBENCH_VERBOSE=1} \
    $exe$quoted"
