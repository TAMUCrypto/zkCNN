aux_source_directory(. conv_src)
list(FILTER conv_src EXCLUDE REGEX "main*")

add_library(cnn_lib ${conv_src})

add_executable(demo_run main_demo.cpp)
target_link_libraries(demo_run cnn_lib hyrax_lib mcl mclbn384_256)