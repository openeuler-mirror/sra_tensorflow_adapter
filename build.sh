#!/bin/bash

# 使用 rpm -qi 提取KML Version 信息
VERSION=$(rpm -qi kml | grep "^Version" | awk '{print $3}')
# 输出 Version
echo "KML Version: $VERSION"
KTFOP_LIB_PATH=/usr/local/sra_inference/lib/neon
KTFOP_INCLUDE_PATH=/usr/local/sra_inference/include
KBLAS_LIB_PATH=""
KBLAS_INCLUDE_PATH=/usr/local/kml/include

if [[ $(printf '%s\n' "$VERSION" "2.5.0" | sort -V | head -n1) == "2.5.0" ]]; then
    KBLAS_LIB_PATH=/usr/local/kml/lib/neon/kblas/locking
else
    KBLAS_LIB_PATH=/usr/local/kml/lib/kblas/locking
fi

if [ -n "$KBLAS_LIB_DIR" ]; then
    KBLAS_LIB_PATH="$KBLAS_LIB_DIR"
fi

echo "ktfop lib path: " $KTFOP_LIB_PATH
echo "kblas lib path: " $KBLAS_LIB_PATH

# install ktfop and kblas
mkdir -p third_party/ktfop/include
mkdir -p third_party/ktfop/lib
cp -a ${KTFOP_LIB_PATH}/libktfop* third_party/ktfop/lib/
cp -a ${KTFOP_INCLUDE_PATH}/ktfop.h third_party/ktfop/include/
cp -a ${KBLAS_LIB_PATH}/* third_party/ktfop/lib/
cp -a ${KBLAS_INCLUDE_PATH}/kblas.h third_party/ktfop/include/

# 配置tensorflow .bazlerc选项
printf "\n\nn\nn\nn\nn\nn\nn\n\nn\n" | ./configure
LD_LIBRARY_PATH=${KTFOP_LIB_PATH}:${KBLAS_LIB_PATH}:$LD_LIBRARY_PATH \
CPLUS_INCLUDE_PATH=${KBLAS_INCLUDE_PATH}:$CPLUS_INCLUDE_PATH \
C_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:$C_INCLUDE_PATH \
bazel build --config=opt --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" --verbose_failures //tensorflow/tools/pip_package:build_pip_package && \
./bazel-bin/tensorflow/tools/pip_package/build_pip_package ./out
