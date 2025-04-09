#/bin/bash
set -x

TF_SERVING_COMPILE_ROOT=$(pwd)/serving
TENSORFLOW_DIR=$(pwd)/tensorflow
DISTDIR=$TF_SERVING_COMPILE_ROOT/download
PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin

PATH=$BAZEL_PATH:$PATH
DISTDIR="${DISTDIR:-$DISTDIR}"
BAZEL_COMPILE_CACHE=$TF_SERVING_COMPILE_ROOT/output

if ! command -v bazel &> /dev/null; then
    echo "Error: Bazel is not installed. Please install Bazel and try again."
    exit 1
fi

bazel version

ENABLE_GCC12=false

FEATURES=$1
IFS=',' read -ra features_array <<< "$FEATURES"
for feature in "${features_array[@]}"; do
    case "$feature" in
        "gcc12")
            ENABLE_GCC12=true
            ;;
        *)
            echo "未识别的特性: $feature"
            ;;
    esac
done

if [ "$ENABLE_GCC12" == true ]; then
    PATH=/opt/openEuler/gcc-toolset-12/root/usr/bin/:$PATH
    LD_LIBRARY_PATH=/opt/openEuler/gcc-toolset-12/root/usr/lib64
    GCC_VERSION=$(gcc -dumpversion | cut -d. -f1)
    if [[ "$GCC_VERSION" != "12" ]]; then
        echo "Error: GCC version is $GCC_VERSION. Please install GCC 12. Consider use command: yum install gcc-toolset-12-gcc*"
        exit 1
    fi
fi

gcc --version
cd $TF_SERVING_COMPILE_ROOT && \
PATH=$PATH \
LD_LIBRARY_PATH=$LD_LIBRARY_PATH \
bazel --output_user_root=$BAZEL_COMPILE_CACHE build -c opt --distdir=$DISTDIR \
--override_repository=org_tensorflow=$TENSORFLOW_DIR \
--copt=-march=armv8.3-a+crc --copt=-O3 --copt=-fprefetch-loop-arrays \
--copt=-Wno-error=maybe-uninitialized --copt=-Werror=stringop-overflow=0 \
tensorflow_serving/model_servers:tensorflow_model_server