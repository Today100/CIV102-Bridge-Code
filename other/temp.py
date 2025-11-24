def calculate_Q(cross_section_dimensions, centroidal_axis_y_bar):
    H = 0
    for y_bottom, w, t in cross_section_dimensions:
        H = max(H, y_bottom + t)


    sorted_dimensions = sorted(cross_section_dimensions, key=lambda x: x[0])

    
    Q_top = 0.0

    for y_bottom, w, t in sorted_dimensions:
        y_top = y_bottom + t
        
        # Check if the segment is partially or fully above the NA
        if y_top > centroidal_axis_y_bar:
            
            # The height of the segment *above* the NA
            h_prime = y_top - max(y_bottom, centroidal_axis_y_bar)
            
            # If there's an area above the NA:
            if h_prime > 0:
                A_prime = w * h_prime
                
                # Centroid of the partial area (A') is at the midpoint of h_prime
                y_c_prime = y_top - (h_prime / 2.0)
                
                # Distance from the centroid of A' to the NA: y_bar' = |y_c_prime - centroidal_axis_y_bar|
                y_bar_prime = abs(y_c_prime - centroidal_axis_y_bar)
                
                Q_top += A_prime * y_bar_prime
                
    ## --- Q FROM THE BOTTOM (Q_bottom) ---
    # We sum segments starting from the bottom (y=0) up to the Neutral Axis (y_bar)
    Q_bottom = 0.0
    
    # Calculate Q for the entire area below the Neutral Axis (NA)
    for y_bottom, w, t in sorted_dimensions:
        y_top = y_bottom + t
        
        # Check if the segment is partially or fully below the NA
        if y_bottom < centroidal_axis_y_bar:
            
            # The height of the segment *below* the NA
            h_prime = min(y_top, centroidal_axis_y_bar) - y_bottom
            
            # If there's an area below the NA:
            if h_prime > 0:
                A_prime = w * h_prime
                
                # Centroid of the partial area (A') is at the midpoint of h_prime
                y_c_prime = y_bottom + (h_prime / 2.0)
                
                # Distance from the centroid of A' to the NA: y_bar' = |y_c_prime - centroidal_axis_y_bar|
                y_bar_prime = abs(y_c_prime - centroidal_axis_y_bar)
                
                Q_bottom += A_prime * y_bar_prime
    
    return Q_top, Q_bottom, H
