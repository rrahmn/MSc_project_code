
import numpy as np
import  matplotlib.pyplot as plt






num_rows, num_cols = 6, 12
chessboard_corners = [(i, j) for i in range(num_rows) for j in range(num_cols)]

# Perspective projection
def project_points_perspective(points, fx=1000, fy=1000, cx=0, cy=0, z=10, rotation_matrix=None, translation_vector=None):
    # Camera intrinsics
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    # If no rotation or translation is provided, use identity and zero translation
    if rotation_matrix is None:
        rotation_matrix = np.eye(3)
    if translation_vector is None:
        translation_vector = np.array([[0], [0], [0]])

    projected_points = []
    for point in points:
        # Convert 2D point to homogenous coordinate
        point_3d = np.array([[point[0]], [point[1]], [1]])

        # Perspective transformation
        point_cam = rotation_matrix @ point_3d + translation_vector
        point_cam /= point_cam[2, 0]  # normalize by z

        # Project to image plane
        point_img = K @ point_cam
        projected_points.append((point_img[0, 0], point_img[1, 0]))

    return projected_points



def distance_from_line(p1, p2, p):
    """Compute the distance from point p to the line defined by points p1 and p2."""
    return abs((p2[1] - p1[1]) * p[0] - (p2[0] - p1[0]) * p[1] + p2[0] * p1[1] - p2[1] * p1[0]) / np.sqrt((p2[1] - p1[1])**2 + (p2[0] - p1[0])**2)

# Function to detect lines that a specified number of points lie on
def find_lines(points, num_points_on_line, tolerance):
    """Find lines that pass through the maximum number of points."""
    lines = []
    for point in points:
        for angle in np.linspace(0, 180, 360):  # Check lines at different angles
            angle_rad = np.radians(angle)
            line_points = [point]

            # Define a distant point on the line
            distant_point = (point[0] + np.cos(angle_rad) * 1000, point[1] + np.sin(angle_rad) * 1000)

            # Count points lying on the line
            for other_point in points:
                if other_point == point:
                    continue
                if distance_from_line(point, distant_point, other_point) < tolerance:
                    line_points.append(other_point)

            # If the number of points on this line is the same as the desired number
            if len(line_points) == num_points_on_line:
                sorted_line = sorted(line_points, key=lambda x: x[0])  # Sort points on the line by x-coordinate
                lines.append((sorted_line[0], sorted_line[-1]))
                break
    return lines






fx = 800  # Focal length in x
fy = 200  # Focal length in y
cx = 150    # Principal point offset in x
cy = 150    # Principal point offset in y

rotation_matrix = np.array([[0.866, -0.5, 0], [0.5, 0.866, 0], [0, 0, 1]])
translation_vector = np.array([[1], [2], [10]])

# Get the projected points using the previously defined project_points_perspective function
projected_points_perspective = project_points_perspective(chessboard_corners, fx,fy,cx,cy, rotation_matrix=rotation_matrix, translation_vector=translation_vector)


# Find the lines on which the points lie using the find_lines function
detected_lines = find_lines(projected_points_perspective, 12, tolerance=10)  # For rows
detected_lines += find_lines(projected_points_perspective, 6, tolerance=10)  # For columns







def find_and_remove_lines_v3(points, num_points_on_line, tolerance, max_iterations=1000):
    """Find lines that pass through the maximum number of points and remove those points."""
    lines = []
    iterations = 0

    while iterations < max_iterations:
        max_line_points = []
        max_line = (None, None)

        for point in points:
            for angle in np.linspace(0, 180, 360):  # Check lines at different angles

                angle_rad = np.radians(angle)
                line_points = [point]

                # Define a distant point on the line
                distant_point = (point[0] + np.cos(angle_rad) * 300, point[1] + np.sin(angle_rad) * 300)

                # Count points lying on the line
                for other_point in points:
                    if other_point == point:
                        continue
                    if distance_from_line(point, distant_point, other_point) < tolerance:
                        line_points.append(other_point)

                # If this line has more points than the previous maximum, update the maximum
                if len(line_points) > len(max_line_points):
                    max_line_points = line_points.copy()
                    sorted_line = sorted(line_points, key=lambda x: x[0])  # Sort points on the line by x-coordinate
                    max_line = (sorted_line[0], sorted_line[-1])

        # If the number of points on the max line is the same as the desired number, add it to the lines
        if len(max_line_points) == num_points_on_line:
            lines.append(max_line)
            # Remove the points that are assigned to this line
            for p in max_line_points:
                points.remove(p)
        else:
            # If no line is found with the desired number of points, break
            break
        iterations += 1

    return lines

# Find the row and column lines again
detected_row_lines = find_and_remove_lines_v3(projected_points_perspective.copy(), 12, tolerance=10)
detected_column_lines = find_and_remove_lines_v3(projected_points_perspective.copy(), 6, tolerance=10)
print(detected_column_lines)

# Visualize the points and the detected row and column lines
plt.figure(figsize=(10,10))
plt.scatter(*zip(*projected_points_perspective), color='blue', marker='o')
for line_start, line_end in detected_row_lines:
    plt.plot([line_start[0], line_end[0]], [line_start[1], line_end[1]], color='red')
for line_start, line_end in detected_column_lines:
    plt.plot([line_start[0], line_end[0]], [line_start[1], line_end[1]], color='green')
plt.title('Points and Detected Row and Column Lines')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.gca().axis('equal')
plt.gca().invert_yaxis()
plt.show()