from shapely.geometry import MultiPoint

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
    
    if min_rect.geom_type == 'Polygon':
        # The result is a Polygon for non-linear point distributions
        rect_coords = list(min_rect.exterior.coords)
        
        # Calculate the lengths of the two unique sides
        # Using squared distances for robustness
        side1_len_sq = (rect_coords[1][0] - rect_coords[0][0])**2 + (rect_coords[1][1] - rect_coords[0][1])**2
        side2_len_sq = (rect_coords[2][0] - rect_coords[1][0])**2 + (rect_coords[2][1] - rect_coords[1][1])**2
        
        major_axis = max(side1_len_sq, side2_len_sq)**0.5
        minor_axis = min(side1_len_sq, side2_len_sq)**0.5
        
        if major_axis == 0:
            return 1.0
        
        return minor_axis / major_axis
    
    elif min_rect.geom_type == 'LineString':
        # The points are collinear, so the result is a LineString
        # Major axis is the length of the line, minor axis is 0
        major_axis = min_rect.length
        minor_axis = 0.0
        
        if major_axis == 0:
            return 1.0
        
        return minor_axis / major_axis
        
    else:
        # Handle single point case from minimum_rotated_rectangle
        return 1.0



if __name__ == "__main__":
        # Example usage
    multipoint_circular = MultiPoint([(0, 0), (1, 0), (0, 1), (1, 1)])
    ellipticity_circular = calculate_ellipticity(multipoint_circular)
    print(f"Ellipticity of square-like points: {ellipticity_circular:.2f}")

    multipoint_elongated = MultiPoint([(0, 0), (5, 0), (10, 0), (15, 0)])
    ellipticity_elongated = calculate_ellipticity(multipoint_elongated)
    print(f"Ellipticity of line-like points: {ellipticity_elongated:.2f}")

    multipoint_scattered = MultiPoint([(1, 1), (5, 2), (2, 5), (6, 6), (3, 3)])
    ellipticity_scattered = calculate_ellipticity(multipoint_scattered)
    print(f"Ellipticity of scattered points: {ellipticity_scattered:.2f}")
