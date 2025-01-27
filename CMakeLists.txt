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
find_package(OpenCV REQUIRED)

find_library(CUDNN_LIBRARY cudnn PATHS /usr/local/cuda/lib64)

include_directories(
  ${CUDA_INCLUDE_DIRS} 
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  /usr/include/eigen3/
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
add_library(NetWork SHARED src/NetWork/NetWork.cpp)
add_library(Mnist SHARED src/Mnist/Mnist.cpp)
cuda_add_library(Loss SHARED src/Loss/Loss.cu)
cuda_add_library(Layer SHARED src/Layer/Activation.cu
                              src/Layer/Add.cu
                              src/Layer/Conv2D.cu
                              src/Layer/Dense.cu
                              src/Layer/Dropout.cu
                              src/Layer/FusedBatchNormalization.cu
                              src/Layer/Layer.cu
                              src/Layer/Lrn.cu
                              src/Layer/Pooling.cu
                              src/Layer/Pad.cu
                              src/Layer/Rnn.cu
                              src/Layer/Softmax.cu)

target_include_directories(Helper PRIVATE include)
target_include_directories(Blob PRIVATE include)
target_include_directories(Loss PRIVATE include)
target_include_directories(Layer PRIVATE include)
target_include_directories(NetWork PRIVATE include)
target_include_directories(Mnist PRIVATE include)

target_link_libraries(Helper ${CUDA_LIBRARIES} ${CUDNN_LIBRARY} cublas)
target_link_libraries(Loss ${CUDA_LIBRARIES} ${CUDNN_LIBRARY} cublas)
target_link_libraries(Blob ${CUDA_LIBRARIES} ${CUDNN_LIBRARY} cublas)
target_link_libraries(Layer ${CUDA_LIBRARIES} ${CUDNN_LIBRARY} cublas Helper Loss Blob)
target_link_libraries(NetWork ${CUDA_LIBRARIES} ${CUDNN_LIBRARY} cublas Layer Helper Blob)
target_link_libraries(Mnist ${OpenCV_LIBRARIES} Blob)

add_executable(mnist_cnn_train example/train/mnist_cnn_train.cpp)
target_link_libraries(mnist_cnn_train Helper Blob Loss Layer NetWork Mnist)
target_include_directories(mnist_cnn_train PRIVATE ${CMAKE_SOURCE_DIR}/include)
ament_target_dependencies(mnist_cnn_train rclcpp)

add_executable(mnist_vgg16_train example/train/mnist_vgg16_train.cpp)
target_link_libraries(mnist_vgg16_train Helper Blob Loss Layer NetWork Mnist)
target_include_directories(mnist_vgg16_train PRIVATE ${CMAKE_SOURCE_DIR}/include)
ament_target_dependencies(mnist_vgg16_train rclcpp)

add_executable(mnist_cnn_test example/test/mnist_cnn_test.cpp)
target_link_libraries(mnist_cnn_test Helper Blob Loss Layer NetWork Mnist)
target_include_directories(mnist_cnn_test PRIVATE ${CMAKE_SOURCE_DIR}/include)
ament_target_dependencies(mnist_cnn_test rclcpp)

add_executable(mnist_rnn_train example/train/mnist_rnn_train.cpp)
target_link_libraries(mnist_rnn_train Helper Blob Loss Layer NetWork Mnist)
target_include_directories(mnist_rnn_train PRIVATE ${CMAKE_SOURCE_DIR}/include)
ament_target_dependencies(mnist_rnn_train rclcpp)

add_executable(mnist_rnn_test example/test/mnist_rnn_test.cpp)
target_link_libraries(mnist_rnn_test Helper Blob Loss Layer NetWork Mnist)
target_include_directories(mnist_rnn_test PRIVATE ${CMAKE_SOURCE_DIR}/include)
ament_target_dependencies(mnist_rnn_test rclcpp)

add_executable(mnist_vgg16_test example/test/mnist_vgg16_test.cpp)
target_link_libraries(mnist_vgg16_test Helper Blob Loss Layer NetWork Mnist)
target_include_directories(mnist_vgg16_test PRIVATE ${CMAKE_SOURCE_DIR}/include)
ament_target_dependencies(mnist_vgg16_test rclcpp)

install(TARGETS
        Helper
        Blob
        Loss
        Layer
        NetWork
        Mnist
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin)

install(TARGETS
  mnist_cnn_train
  mnist_rnn_train
  mnist_vgg16_train
  mnist_cnn_test
  mnist_rnn_test
  mnist_vgg16_test
  DESTINATION lib/${PROJECT_NAME})

ament_package()

install(DIRECTORY include/
        DESTINATION include)
