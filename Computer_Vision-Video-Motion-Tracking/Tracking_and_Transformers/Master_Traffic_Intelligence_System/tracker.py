import math


class Tracker:
    # Step 1: Initialize the tracking system
    def __init__(self):
        # Dictionary to store the center positions (x, y) of objects
        self.center_points = {}
        # Counter to assign unique IDs to new vehicles
        self.id_count = 0

    # Step 2: Update vehicle positions and track IDs
    def update(self, objects_rect):
        """Processes detections and matches them with existing IDs."""
        objects_bbs_ids = []

        for rect in objects_rect:
            x, y, w, h = rect
            # Calculate the center point (cx, cy)
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2

            # Step 3: Check if this vehicle was seen before
            same_object_detected = False
            for id, pt in self.center_points.items():
                # Calculate the distance between the current center and the previous center
                dist = math.hypot(cx - pt[0], cy - pt[1])

                # If the distance is small, it's the same vehicle
                if dist < 35:
                    self.center_points[id] = (cx, cy)
                    objects_bbs_ids.append([x, y, w, h, id])
                    same_object_detected = True
                    break

            # Step 4: If it's a new vehicle, assign a new ID
            if same_object_detected is False:
                self.center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append([x, y, w, h, self.id_count])
                self.id_count += 1

        return objects_bbs_ids