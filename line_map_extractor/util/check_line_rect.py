import numpy as np

def check_line_rect(p1, p2, w, h):
    """
    Check whether a line crosses the image rectangle.
    
    Args:
        p1: 2x1 array, first endpoint pixel coordinate [x, y]
        p2: 2x1 array, second endpoint pixel coordinate [x, y]
        w: int, image width
        h: int, image height
    
    Returns:
        bool: True if line intersects with rectangle, False otherwise
    
    Author: Haodong Jiang <221049033@link.cuhk.edu.cn>
    Version: 1.0
    License: MIT
    """
    
    def is_in(p):
        """Check if point p is inside the rectangle"""
        return 0 <= p[0] <= w and 0 <= p[1] <= h
    
    # Check if either endpoint is inside the rectangle
    if is_in(p1) or is_in(p2):
        return True
    
    # Define the four edges of the rectangle
    # Each edge is defined by two points: [x1, y1, x2, y2]
    edges = [
        [0, 0, 0, h],    # Left edge
        [w, 0, w, h],    # Right edge
        [0, 0, w, 0],    # Top edge
        [0, h, w, h]     # Bottom edge
    ]
    
    for edge in edges:
        a = np.array([edge[0], edge[1]])  # First point of edge
        b = np.array([edge[2], edge[3]])  # Second point of edge
        
        d1 = ccw(p1, p2, a)
        d2 = ccw(p1, p2, b)
        d3 = ccw(a, b, p1)
        d4 = ccw(a, b, p2)
        
        # Check if line segments intersect
        if d1 * d2 < 0 and d3 * d4 < 0:
            return True
        
        # Check for collinear points that lie on the line segment
        if ((d1 == 0 and on_seg(p1, p2, a)) or 
            (d2 == 0 and on_seg(p1, p2, b)) or
            (d3 == 0 and on_seg(a, b, p1)) or
            (d4 == 0 and on_seg(a, b, p2))):
            return True
    
    return False


def ccw(a, b, c):
    """
    Counter-clockwise test for three points.
    Returns positive value if points are counter-clockwise,
    negative if clockwise, and zero if collinear.
    """
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def on_seg(a, b, c):
    """
    Check if point c lies on line segment ab.
    Assumes that a, b, c are collinear.
    """
    return (min(a[0], b[0]) <= c[0] <= max(a[0], b[0]) and
            min(a[1], b[1]) <= c[1] <= max(a[1], b[1]))


# Example usage and test cases
if __name__ == "__main__":
    # Test case 1: Line completely inside rectangle
    p1 = np.array([50, 50])
    p2 = np.array([100, 100])
    w, h = 200, 200
    result = check_line_rect(p1, p2, w, h)
    print(f"Test 1 - Line inside: {result}")  # Should be True
    
    # Test case 2: Line completely outside rectangle
    p1 = np.array([300, 300])
    p2 = np.array([400, 400])
    result = check_line_rect(p1, p2, w, h)
    print(f"Test 2 - Line outside: {result}")  # Should be False
    
    # Test case 3: Line crossing rectangle
    p1 = np.array([-50, 100])
    p2 = np.array([250, 100])
    result = check_line_rect(p1, p2, w, h)
    print(f"Test 3 - Line crossing: {result}")  # Should be True
    
    # Test case 4: Line touching rectangle edge
    p1 = np.array([0, 50])
    p2 = np.array([0, 150])
    result = check_line_rect(p1, p2, w, h)
    print(f"Test 4 - Line touching edge: {result}")  # Should be True 