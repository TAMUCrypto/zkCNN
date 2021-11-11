aux_source_directory(. conv_src)
list(FILTER conv_src EXCLUDE REGEX "main*")

add_library(cnn_lib ${conv_src})

add_executable(demo_vgg_run main_demo_vgg.cpp)
target_link_libraries(demo_vgg_run cnn_lib hyrax_lib mcl mclbn384_256)

add_executable(demo_lenet_run main_demo_lenet.cpp)
target_link_libraries(demo_lenet_run cnn_lib hyrax_lib mcl mclbn384_256)