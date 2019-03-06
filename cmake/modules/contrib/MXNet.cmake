message(STATUS "Build with contrib.mxnet")
file(GLOB MXNET_CONTRIB_SRC "src/contrib/mxnet/*.cc" "src/contrib/mxnet/*.cu")
list(APPEND COMPILER_SRCS ${MXNET_CONTRIB_SRC})
