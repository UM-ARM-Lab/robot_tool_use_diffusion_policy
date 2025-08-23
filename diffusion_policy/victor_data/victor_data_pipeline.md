# Victor Data Pipeline

## Real Robot

### Collection

1. Robot related things use `ros2 bag`
2. Zivid uses local h5 chunks saving

### Processing

1. Convert rosbag file into h5/zarr files
2. Setup timestamps from vision or interpolation
3. Remove plateau (first cheat process joint and gripper data)
4. Register existing transforms
5. Interpolate tf based on recorded tf
6. add/interpolate other values
7. Add images
8. Add finger status from complex finger status
9. Add wrench window data
10. Add progress if used
11. Add extra fields for learning (concat)
12. Add episode ends