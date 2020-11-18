#!/bin/bash
# Copyright 2019 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

set -e
BASEPATH=$(cd "$(dirname $0)"; pwd)
CUDA_PATH=""
export BUILD_PATH="${BASEPATH}/build/"
# print usage message
usage()
{
  echo "Usage:"
  echo "bash build.sh [-d] [-r] [-v] [-c on|off] [-t on|off] [-g on|off] [-h] [-b ge] [-m infer|train] \\"
  echo "              [-a on|off] [-p on|off] [-i] [-L] [-R] [-D on|off] [-j[n]] [-e gpu|ascend|cpu|acl] \\"
  echo "              [-P on|off] [-z [on|off]] [-M on|off] [-V 9.2|10.1] [-I arm64|arm32|x86_64] [-K] \\"
  echo "              [-B on|off] [-E] [-l on|off] [-n full|lite|off] [-T on|off] \\"
  echo "              [-A [cpp|java|object-c] [-C on|off] [-o on|off] [-S on|off] [-k on|off] [-W sse|neon|avx|off] \\"
  echo ""
  echo "Options:"
  echo "    -d Debug mode"
  echo "    -r Release mode, default mode"
  echo "    -v Display build command"
  echo "    -c Enable code coverage, default off"
  echo "    -t Run testcases, default on"
  echo "    -g Use glog to output log, default on"
  echo "    -h Print usage"
  echo "    -b Select other backend, available: \\"
  echo "           ge:graph engine"
  echo "    -m Select graph engine backend mode, available: infer, train, default is infer"
  echo "    -a Enable ASAN, default off"
  echo "    -p Enable pipeline profile, print to stdout, default off"
  echo "    -R Enable pipeline profile, record to json, default off"
  echo "    -i Enable increment building, default off"
  echo "    -L Enable load ANF-IR as input of 'infer', default off"
  echo "    -j[n] Set the threads when building (Default: -j8)"
  echo "    -e Use cpu, gpu, ascend or acl"
  echo "    -P Enable dump anf graph to file in ProtoBuffer format, default on"
  echo "    -D Enable dumping of function graph ir, default on"
  echo "    -z Compile dataset & mindrecord, default on"
  echo "    -n Compile minddata with mindspore lite, available: off, lite, full, lite_cv, full mode in lite train and lite_cv mode in lite predict"
  echo "    -M Enable MPI and NCCL for GPU training, gpu default on"
  echo "    -V Specify the minimum required cuda version, default CUDA 10.1"
  echo "    -I Enable compiling mindspore lite for arm64, arm32 or x86_64, default disable mindspore lite compilation"
  echo "    -K Compile with AKG, default on"
  echo "    -s Enable serving module, default off"
  echo "    -B Enable debugger, default on"
  echo "    -E Enable IBVERBS for parameter server, default off"
  echo "    -l Compile with python dependency, default on"
  echo "    -A Language used by mindspore lite, default cpp"
  echo "    -T Enable on-device training, default off"
  echo "    -C Enable mindspore lite converter compilation, enabled when -I is specified, default on"
  echo "    -o Enable mindspore lite tools compilation, enabled when -I is specified, default on"
  echo "    -S Enable enable download cmake compile dependency from gitee , default off"
  echo "    -k Enable make clean, clean up compilation generated cache "
  echo "    -W Enable x86_64 SSE or AVX instruction set, use [sse|avx|neon|off], default off"
}

# check value of input is 'on' or 'off'
# usage: check_on_off arg_value arg_name
check_on_off()
{
  if [[ "X$1" != "Xon" && "X$1" != "Xoff" ]]; then
    echo "Invalid value $1 for option -$2"
    usage
    exit 1
  fi
}

# check and set options
checkopts()
{
  # Init default values of build options
  THREAD_NUM=8
  DEBUG_MODE="off"
  VERBOSE=""
  ENABLE_COVERAGE="off"
  RUN_TESTCASES="off"
  ENABLE_BACKEND=""
  TRAIN_MODE="INFER"
  ENABLE_ASAN="off"
  ENABLE_PROFILE="off"
  INC_BUILD="off"
  ENABLE_LOAD_IR="off"
  ENABLE_TIMELINE="off"
  ENABLE_DUMP2PROTO="on"
  ENABLE_DUMP_IR="on"
  COMPILE_MINDDATA="on"
  COMPILE_MINDDATA_LITE="lite_cv"
  ENABLE_MPI="off"
  CUDA_VERSION="10.1"
  COMPILE_LITE="off"
  LITE_PLATFORM=""
  SUPPORT_TRAIN="off"
  USE_GLOG="on"
  ENABLE_AKG="on"
  ENABLE_SERVING="off"
  ENABLE_ACL="off"
  ENABLE_DEBUGGER="on"
  ENABLE_IBVERBS="off"
  ENABLE_PYTHON="on"
  ENABLE_GPU="off"
  ENABLE_VERBOSE="off"
  ENABLE_TOOLS="on"
  ENABLE_CONVERTER="on"
  LITE_LANGUAGE="cpp"
  ENABLE_GITEE="off"
  ANDROID_STL="c++_shared"
  ENABLE_MAKE_CLEAN="off"
  X86_64_SIMD="off"

  # Process the options
  while getopts 'drvj:c:t:hsb:a:g:p:ie:m:l:I:LRP:D:zM:V:K:swB:En:T:A:C:o:S:k:W:' opt
  do
    OPTARG=$(echo ${OPTARG} | tr '[A-Z]' '[a-z]')
    case "${opt}" in
      d)
        DEBUG_MODE="on"
        ;;
      n)
        if [[ "X$OPTARG" == "Xoff" || "X$OPTARG" == "Xlite" || "X$OPTARG" == "Xfull" || "X$OPTARG" == "Xlite_cv" ]]; then
          COMPILE_MINDDATA_LITE="$OPTARG"
        else
          echo "Invalid value ${OPTARG} for option -n"
          usage
          exit 1
        fi
        ;;
      r)
        DEBUG_MODE="off"
        ;;
      v)
        ENABLE_VERBOSE="on"
        VERBOSE="VERBOSE=1"
        ;;
      j)
        THREAD_NUM=$OPTARG
        ;;
      c)
        check_on_off $OPTARG c
        ENABLE_COVERAGE="$OPTARG"
        ;;
      t)
        check_on_off $OPTARG t
        RUN_TESTCASES="$OPTARG"
        ;;
      g)
        check_on_off $OPTARG g
        USE_GLOG="$OPTARG"
        ;;
      h)
        usage
        exit 0
        ;;
      b)
        if [[ "X$OPTARG" != "Xge" && "X$OPTARG" != "Xcpu" ]]; then
          echo "Invalid value ${OPTARG} for option -b"
          usage
          exit 1
        fi
        ENABLE_BACKEND=$(echo "$OPTARG" | tr '[a-z]' '[A-Z]')
        if [[ "X$ENABLE_BACKEND" != "XCPU" ]]; then
          ENABLE_CPU="on"
        fi
        ;;
      a)
        check_on_off $OPTARG a
        ENABLE_ASAN="$OPTARG"
        ;;
      p)
        check_on_off $OPTARG p
        ENABLE_PROFILE="$OPTARG"
        ;;
      l)
        check_on_off $OPTARG l
        ENABLE_PYTHON="$OPTARG"
        ;;
      i)
        INC_BUILD="on"
        ;;
      m)
        if [[ "X$OPTARG" != "Xinfer" && "X$OPTARG" != "Xtrain" ]]; then
          echo "Invalid value ${OPTARG} for option -m"
          usage
          exit 1
        fi
        TRAIN_MODE=$(echo "$OPTARG" | tr '[a-z]' '[A-Z]')
        ;;
      L)
        ENABLE_LOAD_IR="on"
        echo "build with enable load anf ir"
        ;;
      R)
        ENABLE_TIMELINE="on"
        echo "enable time_line record"
        ;;
      S)
        check_on_off $OPTARG S
        ENABLE_GITEE="$OPTARG"
        echo "enable download from gitee"
        ;;
      k)
        check_on_off $OPTARG k
        ENABLE_MAKE_CLEAN="$OPTARG"
        echo "enable make clean"
        ;;
      e)
        if [[ "X$OPTARG" == "Xgpu" ]]; then
          ENABLE_GPU="on"
          ENABLE_CPU="on"
          ENABLE_MPI="on"
        elif [[ "X$OPTARG" == "Xd" || "X$OPTARG" == "Xascend" ]]; then
          ENABLE_D="on"
          ENABLE_CPU="on"
          ENABLE_SERVING="on"
        elif [[ "X$OPTARG" == "Xacl" ]]; then
          ENABLE_SERVING="on"
          ENABLE_ACL="on"
        elif [[ "X$OPTARG" == "Xcpu" ]]; then
          ENABLE_CPU="on"
        else
          echo "Invalid value ${OPTARG} for option -e"
          usage
          exit 1
        fi
        ;;
      M)
        check_on_off $OPTARG M
        ENABLE_MPI="$OPTARG"
        ;;
      V)
        if [[ "X$OPTARG" != "X9.2" && "X$OPTARG" != "X10.1" ]]; then
          echo "Invalid value ${OPTARG} for option -V"
          usage
          exit 1
        fi
        if [[ "X$OPTARG" == "X9.2" ]]; then
          echo "Unsupported CUDA version 9.2"
          exit 1
        fi
        CUDA_VERSION="$OPTARG"
        ;;
      P)
        check_on_off $OPTARG p
        ENABLE_DUMP2PROTO="$OPTARG"
        echo "enable dump anf graph to proto file"
        ;;
      D)
        check_on_off $OPTARG D
        ENABLE_DUMP_IR="$OPTARG"
        echo "enable dump function graph ir"
        ;;
      z)
        eval ARG=\$\{$OPTIND\}
        if [[ -n "$ARG" && "$ARG" != -* ]]; then
          OPTARG="$ARG"
          check_on_off $OPTARG z
          OPTIND=$((OPTIND + 1))
        else
          OPTARG=""
        fi
        if [[ "X$OPTARG" == "Xoff" ]]; then
          COMPILE_MINDDATA="off"
        fi
        ;;
      I)
        COMPILE_LITE="on"
        if [[ "$OPTARG" == "arm64" ]]; then
          ENABLE_CONVERTER="off"
          RUN_TESTCASES="on"
          LITE_PLATFORM="arm64"
        elif [[ "$OPTARG" == "arm32" ]]; then
          ENABLE_CONVERTER="off"
          RUN_TESTCASES="on"
          LITE_PLATFORM="arm32"
        elif [[ "$OPTARG" == "x86_64" ]]; then
          ENABLE_CONVERTER="on"
          RUN_TESTCASES="on"
          LITE_PLATFORM="x86_64"
        else
          echo "-I parameter must be arm64、arm32 or x86_64"
          exit 1
        fi
        ;;
      K)
        ENABLE_AKG="on"
        echo "enable compile with akg"
        ;;
      s)
        ENABLE_SERVING="on"
        echo "enable serving"
        ;;
      w)
        ENABLE_SERVING="on"
        echo "enable serving"
        ENABLE_ACL="on"
        echo "enable acl"
        ;;
      B)
        check_on_off $OPTARG B
        ENABLE_DEBUGGER="$OPTARG"
        ;;
      E)
        ENABLE_IBVERBS="on"
        echo "enable IBVERBS for parameter server"
        ;;
      T)
        check_on_off $OPTARG T
        SUPPORT_TRAIN=$OPTARG
        COMPILE_MINDDATA_LITE="full"
        echo "support train on device "
        ;;
      A)
        COMPILE_LITE="on"
        if [[ "$OPTARG" == "cpp" ]]; then
          LITE_LANGUAGE="cpp"
          ANDROID_STL="c++_shared"
        elif [[ "$OPTARG" == "java" ]]; then
          LITE_LANGUAGE="java"
          ENABLE_CONVERTER="off"
          ANDROID_STL="c++_static"
          RUN_TESTCASES="off"
          ENABLE_TOOLS="off"
        elif [[ "$OPTARG" == "object-c" ]]; then
          LITE_LANGUAGE="object-c"
        else
          echo "-A parameter must be cpp, java or object-c"
          exit 1
        fi
        ;;
      C)
        check_on_off $OPTARG C
        ENABLE_CONVERTER="$OPTARG"
        ;;
      o)
        check_on_off $OPTARG o
        ENABLE_TOOLS="$OPTARG"
        ;;
      W)
        if [[ "$OPTARG" != "sse" && "$OPTARG" != "off" && "$OPTARG" != "avx" && "$OPTARG" != "neon" ]]; then
          echo "Invalid value ${OPTARG} for option -W, -W parameter must be sse|neon|avx|off"
          usage
          exit 1
        fi
        if [[ "$OPTARG" == "sse" || "$OPTARG" == "avx" ]]; then
          X86_64_SIMD="$OPTARG"
        fi
        ;;
      *)
        echo "Unknown option ${opt}!"
        usage
        exit 1
    esac
  done
}
checkopts "$@"
echo "---------------- MindSpore: build start ----------------"
mkdir -pv "${BUILD_PATH}/package/mindspore/lib"
git submodule update --init graphengine
if [[ "X$ENABLE_AKG" = "Xon" ]] && [[ "X$ENABLE_D" = "Xon" || "X$ENABLE_GPU" = "Xon" ]]; then
    git submodule update --init --recursive akg
fi

build_exit()
{
    echo "$@" >&2
    stty echo
    exit 1
}

# Create building path
build_mindspore()
{
    echo "start build mindspore project."
    mkdir -pv "${BUILD_PATH}/mindspore"
    cd "${BUILD_PATH}/mindspore"
    CMAKE_ARGS="-DDEBUG_MODE=$DEBUG_MODE -DBUILD_PATH=$BUILD_PATH"
    CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_LOAD_ANF_IR=$ENABLE_LOAD_IR"
    if [[ "X$ENABLE_COVERAGE" = "Xon" ]]; then
      CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_COVERAGE=ON"
    fi
    if [[ "X$RUN_TESTCASES" = "Xon" ]]; then
      CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_TESTCASES=ON"
    fi
    if [[ -n "$ENABLE_BACKEND" ]]; then
      CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_${ENABLE_BACKEND}=ON"
    fi
    if [[ -n "$TRAIN_MODE" ]]; then
        CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_${TRAIN_MODE}=ON"
    fi
    if [[ "X$ENABLE_ASAN" = "Xon" ]]; then
        CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_ASAN=ON"
    fi
    if [[ "X$ENABLE_PROFILE" = "Xon" ]]; then
        CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_PROFILE=ON"
    fi
    if [[ "X$ENABLE_TIMELINE" = "Xon" ]]; then
        CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_TIMELINE=ON"
    fi
    if [[ "X$ENABLE_DUMP2PROTO" = "Xon" ]]; then
        CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_DUMP_PROTO=ON"
    fi
    if [[ "X$ENABLE_GITEE" = "Xon" ]]; then
        CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_GITEE=ON"
    fi
    CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_DUMP_IR=${ENABLE_DUMP_IR}"
    CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_PYTHON=${ENABLE_PYTHON}"
    if [[ "X$ENABLE_MPI" = "Xon" ]]; then
        CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_MPI=ON"
    fi
    if [[ "X$ENABLE_D" = "Xon" ]]; then
        CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_D=ON"
    fi
    if [[ "X$ENABLE_GPU" = "Xon" ]]; then
        CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_GPU=ON -DUSE_CUDA=ON -DCUDA_PATH=$CUDA_PATH -DMS_REQUIRE_CUDA_VERSION=${CUDA_VERSION}"
    fi
    if [[ "X$ENABLE_CPU" = "Xon" ]]; then
        CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_CPU=ON"
    fi
    if [[ "X$COMPILE_MINDDATA" = "Xon" ]]; then
        CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_MINDDATA=ON"
    fi
    if [[ "X$USE_GLOG" = "Xon" ]]; then
        CMAKE_ARGS="${CMAKE_ARGS} -DUSE_GLOG=ON"
    fi
    if [[ "X$ENABLE_AKG" = "Xon" ]] && [[ "X$ENABLE_D" = "Xon" || "X$ENABLE_GPU" = "Xon" ]]; then
        CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_AKG=ON"
    fi
    if [[ "X$ENABLE_SERVING" = "Xon" ]]; then
        CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_SERVING=ON"
    fi
    if [[ "X$ENABLE_ACL" = "Xon" ]]; then
        CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_ACL=ON"
    fi
    if [[ "X$ENABLE_DEBUGGER" = "Xon" ]]; then
        CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_DEBUGGER=ON"
    fi

    if [[ "X$ENABLE_IBVERBS" = "Xon" ]]; then
        CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_IBVERBS=ON"
    fi
    echo "${CMAKE_ARGS}"
    if [[ "X$INC_BUILD" = "Xoff" ]]; then
      cmake ${CMAKE_ARGS} ../..
    fi
    if [[ -n "$VERBOSE" ]]; then
      CMAKE_VERBOSE="--verbose"
    fi
    cmake --build . --target package ${CMAKE_VERBOSE} -j$THREAD_NUM
    echo "success building mindspore project!"
}

checkndk() {
    if [ "${ANDROID_NDK}" ]; then
        echo -e "\e[31mANDROID_NDK_PATH=$ANDROID_NDK  \e[0m"
    else
        echo -e "\e[31mplease set ANDROID_NDK in environment variable for example: export ANDROID_NDK=/root/usr/android-ndk-r20b/ \e[0m"
        exit 1
    fi
}

gene_flatbuffer() {
    FLAT_DIR="${BASEPATH}/mindspore/lite/schema"
    cd ${FLAT_DIR} && rm -rf "${FLAT_DIR}/inner" && mkdir -p "${FLAT_DIR}/inner"
    find . -name "*.fbs" -print0 | xargs -0 "${FLATC}" -c -b
    find . -name "*.fbs" -print0 | xargs -0 "${FLATC}" -c -b --reflect-types --gen-mutable --reflect-names --gen-object-api -o "${FLAT_DIR}/inner"

    FLAT_DIR="${BASEPATH}/mindspore/lite/tools/converter/parser/tflite"
    cd ${FLAT_DIR}
    find . -name "*.fbs" -print0 | xargs -0 "${FLATC}" -c -b --reflect-types --gen-mutable --reflect-names --gen-object-api -o "${FLAT_DIR}/"
}

build_flatbuffer() {
    cd ${BASEPATH}
    FLATC="${BASEPATH}"/third_party/flatbuffers/build/flatc
    if [[ ! -f "${FLATC}" ]]; then
        if [[ "${MSLIBS_SERVER}" ]]; then
            cd "${BASEPATH}"/third_party/
            rm -rf ./v1.11.0.tar.gz ./flatbuffers
            wget http://${MSLIBS_SERVER}:8081/libs/flatbuffers/v1.11.0.tar.gz
            tar -zxvf ./v1.11.0.tar.gz
            mv ./flatbuffers-1.11.0 ./flatbuffers
        else
            git submodule update --init --recursive third_party/flatbuffers
        fi
        cd ${BASEPATH}/third_party/flatbuffers
        rm -rf build && mkdir -pv build && cd build && cmake -DFLATBUFFERS_BUILD_SHAREDLIB=ON .. && make -j$THREAD_NUM
        gene_flatbuffer
    fi
    if [[ "${INC_BUILD}" == "off" ]]; then
        gene_flatbuffer
    fi
}

build_gtest() {
    cd ${BASEPATH}
    git submodule update --init --recursive third_party/googletest
}

gene_clhpp() {
    CL_SRC_DIR="${BASEPATH}/mindspore/lite/src/runtime/kernel/opencl/cl"
    if [ ! -d "${CL_SRC_DIR}" ]; then
      return
    fi
    cd ${CL_SRC_DIR}/
    rm -rf *.inc
    echo "$(cd "$(dirname $0)"; pwd)"
    for file_path in "${CL_SRC_DIR}"/*
    do
        file="$(basename ${file_path})"
        inc_file=$(echo ${CL_SRC_DIR}/${file} | sed 's/$/.inc/')
        sed 's/\\/\\\\/g;s/\"/\\\"/g;s/^/\"/;s/$/\\n\" \\/' ${CL_SRC_DIR}/${file} > ${inc_file}
        kernel_name=$(echo ${file} | sed s'/.\{3\}$//')
        sed -i "1i\static const char *${kernel_name}_source =\"\\n\" \\" ${inc_file}
        sed -i '$a\;' ${inc_file}
    done
}

gene_ocl_program() {
    OCL_SRC_DIR="${BASEPATH}/mindspore/lite/src/runtime/kernel/opencl/cl"
    SPIRV_DIR=build/spirv
    [ -n "${SPIRV_DIR}" ] && rm -rf ${SPIRV_DIR}
    mkdir -pv ${SPIRV_DIR}
    if [ ! -d "${OCL_SRC_DIR}" ]; then
      return
    fi
    for file_path in "${OCL_SRC_DIR}"/*
    do
      ocl_file="$(basename ${file_path})"
      if [ "${ocl_file##*.}" != "cl" ]; then
        continue
      fi
      clang -Xclang -finclude-default-header -cl-std=CL2.0 --target=spir64-unknown-unknown -emit-llvm \
            -c -O0 -o ${SPIRV_DIR}/${ocl_file%.*}.bc ${OCL_SRC_DIR}/${ocl_file}
    done

    bcs=$(ls ${SPIRV_DIR}/*.bc)
    llvm-link ${bcs} -o ${SPIRV_DIR}/program.bc
    llvm-spirv -o ${SPIRV_DIR}/program.spv ${SPIRV_DIR}/program.bc

    CL_PROGRAM_PATH="${BASEPATH}/mindspore/lite/src/runtime/kernel/opencl/cl/program.inc"
    echo "#include <vector>" > ${CL_PROGRAM_PATH}
    echo "std::vector<unsigned char> g_program_binary = {" >> ${CL_PROGRAM_PATH}
    #hexdump -v -e '16/1 "0x%02x, " "\n"' ${SPIRV_DIR}/program.spv >> ${CL_PROGRAM_PATH}
    hexdump -v -e '1/1 "0x%02x, "' ${SPIRV_DIR}/program.spv >> ${CL_PROGRAM_PATH}
    echo "};" >> ${CL_PROGRAM_PATH}
    echo "Compile SPIRV done"
}

build_opencl() {
    cd ${BASEPATH}
    git submodule update --init third_party/OpenCL-Headers
    git submodule update --init third_party/OpenCL-CLHPP
    if [[ "${OPENCL_OFFLINE_COMPILE}" == "on" ]]; then
        gene_ocl_program
    else
        gene_clhpp
    fi
}

build_opencv() {
    # check what platform we are building opencv on
    cd ${BASEPATH}
    if [[ "${LITE_PLATFORM}" == "x86_64" ]]; then
        OPENCV_BIN="${BASEPATH}"/third_party/opencv/build/lib/libopencv_core.so.4.2.0
    elif [[ "${LITE_PLATFORM}" == "arm32" ]]; then
        OPENCV_BIN="${BASEPATH}"/third_party/opencv/build/lib/armeabi-v7a/libopencv_core.so
    else
        OPENCV_BIN="${BASEPATH}"/third_party/opencv/build/lib/arm64-v8a/libopencv_core.so

    fi
    if [[ ! -f "${OPENCV_BIN}" ]]; then
        if [[ "${MSLIBS_SERVER}" ]]; then
	    cd "${BASEPATH}"/third_party/
	    rm -rf 4.2.0.tar.gz ./opencv
	    wget http://${MSLIBS_SERVER}:8081/libs/opencv/4.2.0.tar.gz
            tar -zxvf ./4.2.0.tar.gz
	    mv ./opencv-4.2.0 ./opencv
            rm -rf 4.2.0.tar.gz
	else
            git submodule update --init --recursive third_party/opencv
        fi
        cd ${BASEPATH}/third_party/opencv
        rm -rf build && mkdir -p build && cd build && cmake ${CMAKE_MINDDATA_ARGS} -DBUILD_SHARED_LIBS=ON -DBUILD_ANDROID_PROJECTS=OFF \
          -DBUILD_LIST=core,imgcodecs,imgproc -DBUILD_ZLIB=ON .. && make -j$THREAD_NUM
    fi
}

build_jpeg_turbo() {
    if [ -d  "${BASEPATH}"/third_party/libjpeg-turbo/lib ];then
        rm -rf "${BASEPATH}"/third_party/libjpeg-turbo/lib
    fi
    cd ${BASEPATH}
    if [[ "${LITE_PLATFORM}" == "x86_64" ]]; then
        JPEG_TURBO="${BASEPATH}"/third_party/libjpeg-turbo/lib/libjpeg.so.62.3.0
    else
        JPEG_TURBO="${BASEPATH}"/third_party/libjpeg-turbo/lib/libjpeg.so
    fi

    if [[ ! -f "${JPEG_TURBO}" ]]; then
        if [[ "${MSLIBS_SERVER}" ]]; then
            cd "${BASEPATH}"/third_party/
	    rm -rf 2.0.4.tar.gz ./libjpeg-turbo
	    wget http://${MSLIBS_SERVER}:8081/libs/jpeg_turbo/2.0.4.tar.gz
	    tar -zxvf ./2.0.4.tar.gz
	    mv ./libjpeg-turbo-2.0.4 ./libjpeg-turbo
	    rm -rf ./2.0.4.tar.gz
        else
            git submodule update --init --recursive third_party/libjpeg-turbo
        fi

        cd ${BASEPATH}/third_party/libjpeg-turbo
        rm -rf build && mkdir -p build && cd build && cmake ${CMAKE_MINDDATA_ARGS} -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_INSTALL_PREFIX="${BASEPATH}/third_party/libjpeg-turbo" .. && make -j$THREAD_NUM && make install
    fi
}

build_eigen() {
    cd ${BASEPATH}
    if [[ "${MSLIBS_SERVER}" ]]; then
        cd "${BASEPATH}"/third_party/
	rm -rf ./eigen-3.*.tar.gz ./eigen
        wget http://${MSLIBS_SERVER}:8081/libs/eigen3/eigen-3.3.7.tar.gz
        tar -zxvf ./eigen-3.3.7.tar.gz
        mv ./eigen-3.3.7 ./eigen
	rm -rf ./eigen-3.*.tar.gz
    else
        git submodule update --init --recursive third_party/eigen

    fi
}

build_minddata_lite_deps()
{
  echo "start build minddata lite project"
  if [[ "${LITE_PLATFORM}" == "arm64" ]]; then
      CMAKE_MINDDATA_ARGS="-DCMAKE_TOOLCHAIN_FILE=${ANDROID_NDK}/build/cmake/android.toolchain.cmake -DANDROID_NATIVE_API_LEVEL=19    \
            -DANDROID_NDK=${ANDROID_NDK} -DANDROID_ABI=arm64-v8a -DANDROID_TOOLCHAIN_NAME=aarch64-linux-android-clang                 \
            -DANDROID_STL=c++_shared -DCMAKE_BUILD_TYPE=${BUILD_TYPE}"
  elif [[ "${LITE_PLATFORM}" == "arm32" ]]; then
      CMAKE_MINDDATA_ARGS="-DCMAKE_TOOLCHAIN_FILE=${ANDROID_NDK}/build/cmake/android.toolchain.cmake -DANDROID_NATIVE_API_LEVEL=19    \
            -DANDROID_NDK=${ANDROID_NDK} -DANDROID_ABI=armeabi-v7a -DANDROID_TOOLCHAIN_NAME=clang                                     \
            -DANDROID_STL=c++_shared -DCMAKE_BUILD_TYPE=${BUILD_TYPE}"
  else
      CMAKE_MINDDATA_ARGS="-DCMAKE_BUILD_TYPE=${BUILD_TYPE}"
  fi
  build_eigen
  build_jpeg_turbo
}

get_version() {
    VERSION_MAJOR=$(grep "const int ms_version_major =" ${BASEPATH}/mindspore/lite/include/version.h | tr -dc "[0-9]")
    VERSION_MINOR=$(grep "const int ms_version_minor =" ${BASEPATH}/mindspore/lite/include/version.h | tr -dc "[0-9]")
    VERSION_REVISION=$(grep "const int ms_version_revision =" ${BASEPATH}/mindspore/lite/include/version.h | tr -dc "[0-9]")
    VERSION_STR=${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_REVISION}
}

build_lite()
{
    get_version
    echo "============ Start building MindSpore Lite ${VERSION_STR} ============"
    if [ "${ENABLE_GPU}" == "on" ] && [ "${LITE_PLATFORM}" == "arm64" ]; then
      echo "start build opencl"
      build_opencl
    fi
    if [ "${RUN_TESTCASES}" == "on" ]; then
        build_gtest
    fi

    if [ "${COMPILE_MINDDATA_LITE}" == "lite" ] || [ "${COMPILE_MINDDATA_LITE}" == "full" ]; then
        build_minddata_lite_deps
    fi

    cd "${BASEPATH}/mindspore/lite"
    if [[ "${INC_BUILD}" == "off" ]]; then
        rm -rf build
    fi
    mkdir -pv build
    cd build
    BUILD_TYPE="Release"
    if [[ "${DEBUG_MODE}" == "on" ]]; then
      BUILD_TYPE="Debug"
    fi

    if [[ "${LITE_PLATFORM}" == "arm64" ]]; then
        checkndk
        cmake -DCMAKE_TOOLCHAIN_FILE="${ANDROID_NDK}/build/cmake/android.toolchain.cmake" -DANDROID_NATIVE_API_LEVEL="19"      \
              -DANDROID_NDK="${ANDROID_NDK}" -DANDROID_ABI="arm64-v8a" -DANDROID_TOOLCHAIN_NAME="aarch64-linux-android-clang"  \
              -DANDROID_STL=${ANDROID_STL} -DCMAKE_BUILD_TYPE=${BUILD_TYPE} -DSUPPORT_TRAIN=${SUPPORT_TRAIN}                     \
              -DPLATFORM_ARM64=on -DENABLE_NEON=on -DENABLE_FP16="off"      \
              -DENABLE_TOOLS=${ENABLE_TOOLS} -DENABLE_CONVERTER=${ENABLE_CONVERTER} -DBUILD_TESTCASES=${RUN_TESTCASES} \
              -DSUPPORT_GPU=${ENABLE_GPU} -DOFFLINE_COMPILE=${OPENCL_OFFLINE_COMPILE} -DBUILD_MINDDATA=${COMPILE_MINDDATA_LITE} \
              -DCMAKE_INSTALL_PREFIX=${BASEPATH}/output/tmp -DMS_VERSION_MAJOR=${VERSION_MAJOR}                           \
              -DMS_VERSION_MINOR=${VERSION_MINOR} -DMS_VERSION_REVISION=${VERSION_REVISION} -DENABLE_VERBOSE=${ENABLE_VERBOSE} \
              "${BASEPATH}/mindspore/lite"
    elif [[ "${LITE_PLATFORM}" == "arm32" ]]; then
        checkndk
        cmake -DCMAKE_TOOLCHAIN_FILE="${ANDROID_NDK}/build/cmake/android.toolchain.cmake" -DANDROID_NATIVE_API_LEVEL="19"      \
              -DANDROID_NDK="${ANDROID_NDK}" -DANDROID_ABI="armeabi-v7a" -DANDROID_TOOLCHAIN_NAME="clang"                      \
              -DANDROID_STL=${ANDROID_STL}  -DCMAKE_BUILD_TYPE=${BUILD_TYPE}                                                      \
              -DPLATFORM_ARM32=on -DENABLE_NEON=on -DSUPPORT_TRAIN=${SUPPORT_TRAIN}  \
              -DENABLE_TOOLS=${ENABLE_TOOLS} -DENABLE_CONVERTER=${ENABLE_CONVERTER} -DBUILD_TESTCASES=${RUN_TESTCASES} \
              -DSUPPORT_GPU=${ENABLE_GPU} -DOFFLINE_COMPILE=${OPENCL_OFFLINE_COMPILE} -DBUILD_MINDDATA=${COMPILE_MINDDATA_LITE} \
              -DCMAKE_INSTALL_PREFIX=${BASEPATH}/output/tmp -DMS_VERSION_MAJOR=${VERSION_MAJOR}                           \
              -DMS_VERSION_MINOR=${VERSION_MINOR} -DMS_VERSION_REVISION=${VERSION_REVISION} -DENABLE_VERBOSE=${ENABLE_VERBOSE} \
               "${BASEPATH}/mindspore/lite"
    else
        cmake -DPLATFORM_ARM64=off -DSUPPORT_TRAIN=${SUPPORT_TRAIN}   \
        -DENABLE_TOOLS=${ENABLE_TOOLS} -DENABLE_CONVERTER=${ENABLE_CONVERTER} -DBUILD_TESTCASES=${RUN_TESTCASES} \
        -DCMAKE_BUILD_TYPE=${BUILD_TYPE} -DSUPPORT_GPU=${ENABLE_GPU} -DBUILD_MINDDATA=${COMPILE_MINDDATA_LITE} \
        -DOFFLINE_COMPILE=${OPENCL_OFFLINE_COMPILE} -DCMAKE_INSTALL_PREFIX=${BASEPATH}/output/tmp  \
        -DMS_VERSION_MAJOR=${VERSION_MAJOR} -DMS_VERSION_MINOR=${VERSION_MINOR} -DMS_VERSION_REVISION=${VERSION_REVISION} \
        -DENABLE_VERBOSE=${ENABLE_VERBOSE} -DX86_64_SIMD=${X86_64_SIMD} "${BASEPATH}/mindspore/lite"
    fi
    make -j$THREAD_NUM && make install && make package
    COMPILE_RET=$?

    if [[ "${COMPILE_RET}" -ne 0 ]]; then
        echo "---------------- mindspore lite: build failed ----------------"
        exit 1
    else
        mv ${BASEPATH}/output/tmp/*.tar.gz* ${BASEPATH}/output/
        rm -rf ${BASEPATH}/output/tmp/
        echo "---------------- mindspore lite: build success ----------------"
        if [[ "X$LITE_LANGUAGE" = "Xcpp" ]]; then
            exit 0
        fi
    fi
}

build_lite_java_arm64() {
    # build mindspore-lite arm64
    if [[ "X$INC_BUILD" = "Xoff" ]] || [[ ! -f "${BASEPATH}/output/mindspore-lite-${VERSION_STR}-runtime-arm64-cpu.tar.gz" ]]; then
      LITE_PLATFORM="arm64"
      INC_BUILD_COPY=${INC_BUILD}
      INC_BUILD="off"
      build_lite
      INC_BUILD=${INC_BUILD_COPY}
    fi
    # copy arm64 so
    cd ${BASEPATH}/output/
    rm -rf mindspore-lite-${VERSION_STR}-runtime-arm64-cpu
    tar -zxvf mindspore-lite-${VERSION_STR}-runtime-arm64-cpu.tar.gz
    [ -n "${JAVA_PATH}" ] && rm -rf ${JAVA_PATH}/java/app/libs/arm64-v8a/
    mkdir -p ${JAVA_PATH}/java/app/libs/arm64-v8a/
    cp ${BASEPATH}/output/mindspore-lite-${VERSION_STR}-runtime-arm64-cpu/lib/libmindspore-lite.so ${JAVA_PATH}/java/app/libs/arm64-v8a/
    echo mindspore-lite-${VERSION_STR}-runtime-arm64-cpu
    [ -n "${VERSION_STR}" ] && rm -rf mindspore-lite-${VERSION_STR}-runtime-arm64-cpu
}

build_lite_java_arm32() {
    # build mindspore-lite arm32
    if [[ "X$INC_BUILD" = "Xoff" ]] || [[ ! -f "${BASEPATH}/output/mindspore-lite-${VERSION_STR}-runtime-arm32-cpu.tar.gz" ]]; then
      LITE_PLATFORM="arm32"
      INC_BUILD_COPY=${INC_BUILD}
      INC_BUILD="off"
      build_lite
      INC_BUILD=${INC_BUILD_COPY}
    fi
    # copy arm32 so
    cd ${BASEPATH}/output/
    rm -rf mindspore-lite-${VERSION_STR}-runtime-arm32-cpu
    tar -zxvf mindspore-lite-${VERSION_STR}-runtime-arm32-cpu.tar.gz
    [ -n "${JAVA_PATH}" ] && rm -rf ${JAVA_PATH}/java/app/libs/armeabi-v7a/
    mkdir -p ${JAVA_PATH}/java/app/libs/armeabi-v7a/
    cp ${BASEPATH}/output/mindspore-lite-${VERSION_STR}-runtime-arm32-cpu/lib/libmindspore-lite.so ${JAVA_PATH}/java/app/libs/armeabi-v7a/
    [ -n "${VERSION_STR}" ] && rm -rf mindspore-lite-${VERSION_STR}-runtime-arm32-cpu
}

build_jni_arm64() {
    # build jni so
    cd "${BASEPATH}/mindspore/lite/build"
    rm -rf java
    mkdir -pv java
    cd java
    cmake -DCMAKE_TOOLCHAIN_FILE="${ANDROID_NDK}/build/cmake/android.toolchain.cmake" -DANDROID_NATIVE_API_LEVEL="19"      \
          -DANDROID_NDK="${ANDROID_NDK}" -DANDROID_ABI="arm64-v8a" -DANDROID_TOOLCHAIN_NAME="aarch64-linux-android-clang"  \
          -DMS_VERSION_MAJOR=${VERSION_MAJOR} -DMS_VERSION_MINOR=${VERSION_MINOR} -DMS_VERSION_REVISION=${VERSION_REVISION} \
          -DANDROID_STL="c++_static" -DCMAKE_BUILD_TYPE=${BUILD_TYPE} -DENABLE_VERBOSE=${ENABLE_VERBOSE} \
          -DPLATFORM_ARM64=on "${JAVA_PATH}/java/app/src/main/native"
    make -j$THREAD_NUM
    COMPILE_RET=$?
    if [[ "${COMPILE_RET}" -ne 0 ]]; then
        echo "---------------- mindspore lite: build jni arm64 failed----------------"
        exit 1
    fi
    mkdir -p ${JAVA_PATH}/java/app/libs/arm64-v8a/
    cp ${BASEPATH}/mindspore/lite/build/java/libmindspore-lite-jni.so ${JAVA_PATH}/java/app/libs/arm64-v8a/
}

build_jni_arm32() {
    # build jni so
    cd "${BASEPATH}/mindspore/lite/build"
    rm -rf java
    mkdir -pv java
    cd java
    cmake -DCMAKE_TOOLCHAIN_FILE="${ANDROID_NDK}/build/cmake/android.toolchain.cmake" -DANDROID_NATIVE_API_LEVEL="19"      \
          -DANDROID_NDK="${ANDROID_NDK}" -DANDROID_ABI="armeabi-v7a" -DANDROID_TOOLCHAIN_NAME="aarch64-linux-android-clang"  \
          -DMS_VERSION_MAJOR=${VERSION_MAJOR} -DMS_VERSION_MINOR=${VERSION_MINOR} -DMS_VERSION_REVISION=${VERSION_REVISION} \
          -DANDROID_STL="c++_static" -DCMAKE_BUILD_TYPE=${BUILD_TYPE} -DENABLE_VERBOSE=${ENABLE_VERBOSE} \
          -DPLATFORM_ARM32=on "${JAVA_PATH}/java/app/src/main/native"
    make -j$THREAD_NUM
    COMPILE_RET=$?
    if [[ "${COMPILE_RET}" -ne 0 ]]; then
        echo "---------------- mindspore lite: build jni arm32 failed----------------"
        exit 1
    fi
    mkdir -p ${JAVA_PATH}/java/app/libs/armeabi-v7a/
    cp ${BASEPATH}/mindspore/lite/build/java/libmindspore-lite-jni.so ${JAVA_PATH}/java/app/libs/armeabi-v7a/
}

build_java() {
  JAVA_PATH=${BASEPATH}/mindspore/lite/java
  get_version
  build_lite_java_arm64
  build_lite_java_arm32
  build_jni_arm64
  build_jni_arm32

  # build aar
  ## check sdk gradle
  cd ${JAVA_PATH}/java
  rm -rf .gradle build gradle gradlew gradlew.bat build app/build

  gradle init
  gradle wrapper
  ./gradlew build

  gradle publish -PLITE_VERSION=${VERSION_STR}

  cd ${JAVA_PATH}/java/app/build
  zip -r mindspore-lite-maven-${VERSION_STR}.zip mindspore
  # copy output
  cp mindspore-lite-maven-${VERSION_STR}.zip ${BASEPATH}/output/
  exit 0
}

make_clean()
{
  echo "enbale make clean"
  cd "${BUILD_PATH}/mindspore"
  cmake --build . --target clean
}

if [[ "X$COMPILE_LITE" = "Xon" ]]; then
  if [[ "X$LITE_LANGUAGE" = "Xjava" ]]; then
    build_java
  else
    build_lite
  fi
else
    build_mindspore
fi

if [[ "X$ENABLE_MAKE_CLEAN" = "Xon" ]]; then
  make_clean
fi

cp -rf ${BUILD_PATH}/package/mindspore/lib ${BUILD_PATH}/../mindspore
cp -rf ${BUILD_PATH}/package/mindspore/*.so ${BUILD_PATH}/../mindspore

echo "---------------- mindspore: build end   ----------------"
