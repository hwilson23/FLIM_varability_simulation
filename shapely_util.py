from shapely.geometry import MultiPoint, Polygon

def calculate_ellipticity(multipoint):
    """
    Calculates the ellipticity of a MultiPoint geometry.
    Ellipticity is approximated using the ratio of the minor and major axes
    of the minimum rotated bounding rectangle.
    """
    if len(multipoint.geoms) < 2:
        # A single point is perfectly circular
        return 1.0
        
    min_rect = multipoint.minimum_rotated_rectangle
    
    # Get the coordinates of the rectangle's vertices
    rect_coords = list(min_rect.exterior.coords)

    # Calculate the lengths of the two unique sides
    # We use distance() method from Shapely for accuracy
    from shapely.geometry import Point
    side1_len = Point(rect_coords[0]).distance(Point(rect_coords[1]))
    side2_len = Point(rect_coords[1]).distance(Point(rect_coords[2]))

    # Determine major and minor axis lengths
    major_axis = max(side1_len, side2_len)
    minor_axis = min(side1_len, side2_len)

    if major_axis == 0:
        return 1.0
    
    return minor_axis / major_axis

if __name__ == "__main__":
    multipoint_circular = MultiPoint([(0, 0), (1, 0), (0, 1), (1, 1)])
    ellipticity_circular = calculate_ellipticity(multipoint_circular)
    print(f"Ellipticity of square-like points: {ellipticity_circular:.2f}")

    multipoint_elongated = MultiPoint([(0, 0), (5, 0), (10, 0), (15, 0)])
    ellipticity_elongated = calculate_ellipticity(multipoint_elongated)
    print(f"Ellipticity of line-like points: {ellipticity_elongated:.2f}")