#!/bin/bash

# 配置tensorflow .bazlerc选项
printf "\n\nn\nn\nn\nn\nn\nn\n\nn\n" | ./configure

# install ktfop
mkdir -p third_party/ktfop/include
mkdir -p third_party/ktfop/lib
cp -a /usr/local/sra_inference/lib/neon/libktfop* third_party/ktfop/lib/
cp -a /usr/local/sra_inference/include/* third_party/ktfop/include/

# 使用 rpm -qi 提取KML Version 信息
VERSION=$(rpm -qi kml | grep "^Version" | awk '{print $3}')
# 输出 Version
echo "KML Version: $VERSION"

# 比较版本号是否 >= 2.5.0
if [[ $(printf '%s\n' "$VERSION" "2.5.0" | sort -V | head -n1) == "2.5.0" ]]; then 
    LD_LIBRARY_PATH=/usr/local/sra_inference/lib/neon:/usr/local/kml/lib/neon/kblas/locking/:$LD_LIBRARY_PATH \
    CPLUS_INCLUDE_PATH=/usr/local/kml/include:$CPLUS_INCLUDE_PATH \
    C_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:$C_INCLUDE_PATH \
    bazel build --config=opt --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" --verbose_failures //tensorflow/tools/pip_package:build_pip_package && \
    ./bazel-bin/tensorflow/tools/pip_package/build_pip_package ./out
else
    LD_LIBRARY_PATH=/usr/local/sra_inference/lib/neon:/usr/local/kml/lib/kblas/locking/:$LD_LIBRARY_PATH \
    CPLUS_INCLUDE_PATH=/usr/local/kml/include:$CPLUS_INCLUDE_PATH \
    C_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:$C_INCLUDE_PATH \
    bazel build --config=opt --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" --verbose_failures //tensorflow/tools/pip_package:build_pip_package && \
    ./bazel-bin/tensorflow/tools/pip_package/build_pip_package ./out
fi
