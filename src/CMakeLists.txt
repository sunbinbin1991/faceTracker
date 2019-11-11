
include_directories(${OpenCV_INCLUDE_DIRS})

#add detect related
include_directories(./detect)

add_library(mtdetect ./detect/mtcnn.cpp)

#add landrmark related
include_directories(./landmark)

add_library(landmark ./landmark/landmark.cpp)

#add track related
include_directories(./track)

add_library(track ./track/ctracker.cpp ./track/track.cpp)