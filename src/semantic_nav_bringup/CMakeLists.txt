cmake_minimum_required(VERSION 3.8)
project(semantic_nav_bringup)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

# Pure-Python helpers (world_tools) importable as semantic_nav_bringup.*
ament_python_install_package(${PROJECT_NAME})

install(
  DIRECTORY launch config worlds rviz urdf models maps
  DESTINATION share/${PROJECT_NAME}
)

install(
  PROGRAMS
    scripts/extract_ground_truth.py
    scripts/generate_house_world.py
    scripts/generate_local_models.py
    scripts/measure_sim_performance.py
    scripts/mapping_drive.py
    scripts/nav_goal_test.py
    scripts/detection_tour.py
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()

  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(test_world_tools test/test_world_tools.py TIMEOUT 60)
  ament_add_pytest_test(test_house_world test/test_house_world.py TIMEOUT 60)
  ament_add_pytest_test(test_nav_utils test/test_nav_utils.py TIMEOUT 60)
endif()

ament_package()
