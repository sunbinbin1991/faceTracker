
add_executable(test_detect test_detect.cpp )

target_link_libraries(test_detect mtdetect ncnn ${OpenCV_LIBS})


add_executable(test_landmark test_landmark.cpp )

target_link_libraries(test_landmark mtdetect landmark ncnn ${OpenCV_LIBS})



