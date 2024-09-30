cmake_minimum_required(VERSION 3.8)
project(mnist_cudnn LANGUAGES CXX CUDA)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(CUDA REQUIRED)

include_directories(
  ${CUDA_INCLUDE_DIRS} 
  include
)

link_directories(/usr/local/cuda/lib64)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # uncomment the line when a copyright and license is not present in all source files
  #set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # uncomment the line when this package is not in a git repo
  #set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

add_library(Helper SHARED src/Helper/Helper.cpp)
add_library(Blob SHARED src/Blob/Blob.cpp)
cuda_add_library(Loss SHARED src/Loss/Loss.cu)

target_include_directories(Helper PUBLIC include)
target_include_directories(Blob PUBLIC include)
target_include_directories(Loss PUBLIC include)

add_executable(mnist_train src/mnist_train.cpp)
target_link_libraries(mnist_train ${CUDA_LIBRARIES} cublas cudnn Helper Blob Loss)
target_include_directories(mnist_train PRIVATE ${CMAKE_SOURCE_DIR}/include)
ament_target_dependencies(mnist_train rclcpp)

install(TARGETS
        Helper
        Blob
        Loss
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin)

install(TARGETS
  mnist_train
  DESTINATION lib/${PROJECT_NAME})

ament_package()

install(DIRECTORY include/
        DESTINATION include)
