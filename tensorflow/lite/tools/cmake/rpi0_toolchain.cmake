set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR armv6)


set(ARMCC_ROOT ${CMAKE_CURRENT_LIST_DIR}/toolchains/arm-rpi-linux-gnueabihf/x64-gcc-6.5.0/arm-rpi-linux-gnueabihf)
set(ARMCC_PREFIX ${ARMCC_ROOT}/bin/arm-rpi-linux-gnueabihf-)

set(ARMCC_FLAGS "-march=armv6 -mfpu=vfp -funsafe-math-optimizations")
set(ARMCC_FLAGS "${ARMCC_FLAGS} -isystem ${ARMCC_ROOT}/lib/gcc/arm-rpi-gnueabihf/6.5.0/include")
set(ARMCC_FLAGS "${ARMCC_FLAGS} -isystem ${ARMCC_ROOT}/lib/gcc/arm-rpi-gnueabihf/6.5.0/include-fixed")
set(ARMCC_FLAGS "${ARMCC_FLAGS} -isystem ${ARMCC_ROOT}/arm-rpi-linux-gnueabihf/include/c++/6.5.0")
set(ARMCC_FLAGS "${ARMCC_FLAGS} -isystem ${ARMCC_ROOT}/arm-rpi-linux-gnueabihf/sysroot/usr/include")
set(ARMCC_FLAGS "${ARMCC_FLAGS} -isystem /usr/include/python3.7m")
#set(ARMCC_FLAGS "${ARMCC_FLAGS} -isystem /usr/include")

set(CMAKE_C_COMPILER ${ARMCC_PREFIX}gcc CACHE INTERNAL "")
set(CMAKE_CXX_COMPILER ${ARMCC_PREFIX}g++ CACHE INTERNAL "")


set(CMAKE_C_FLAGS ${ARMCC_FLAGS} CACHE INTERNAL "")
set(CMAKE_CXX_FLAGS ${ARMCC_FLAGS}  CACHE INTERNAL "")