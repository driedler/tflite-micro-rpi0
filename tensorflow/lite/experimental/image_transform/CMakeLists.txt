
add_library(image_transform)


target_include_directories(image_transform
PRIVATE 
    .
)

target_sources(image_transform
PRIVATE 
    get_transform.cc
    transform_image.cc
    standardize_mean_and_std.cc
)