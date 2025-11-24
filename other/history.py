import matplotlib.pyplot as plt

def plot_sfd_bmd(train_1_weight, train_2_weight, train_3_weight, train_1_location=1028):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    train_weight = {
    1: train_1_weight,
    2: train_2_weight,
    3: train_3_weight
    }

    # SET UP
    n_trains = 3

    """
                        _   T3    _         _   T2    _         _  T1     _
    __________._________|_________|_________|_________|_________|_________|__________.
    |<- 120 ->|<- 52 ->|<- 176 ->|<- 164 ->|<- 176 ->|<- 164 ->|<- 176 ->|<- 52 ->|<- 120 ->|
    """

    front_train_spacing = 52
    wheel_spacing = 176
    train_spacing = 164

    point_load_pair = {}

    for num in range(n_trains):
        wheel_1_location = train_1_location - num*train_spacing - num*wheel_spacing
        wheel_1_weight = train_weight[num+1]/2

        wheel_2_location = wheel_1_location - wheel_spacing
        wheel_2_weight = train_weight[num+1]/2

        if wheel_1_location >= 0 and wheel_1_location <= 1200:
            point_load_pair[wheel_1_location] = wheel_1_weight
        if wheel_2_location >= 0 and wheel_2_location <= 1200:
            point_load_pair[wheel_2_location] = wheel_2_weight

    total_load = sum(point_load_pair.values())
    # print(point_load_pair)


    total_negative_moment_at_0 = 0
    for location, weight in point_load_pair.items():
        total_negative_moment_at_0 -= weight * location

    # print(total_negative_moment_at_0)


    reaction_force_2 = -total_negative_moment_at_0 / 1200
    reaction_force_1 = total_load - reaction_force_2

    point_forces_location = [0]
    point_forces = [reaction_force_1]

    for location in sorted(point_load_pair.keys()):
        point_forces_location.append(location)
        point_forces.append(-point_load_pair[location])

    point_forces_location.append(1200)
    point_forces.append(reaction_force_2)

    print("****Point Force Locations and Forces****")
    for pl, pf in zip(point_forces_location, point_forces):
        print(f"Location: {pl}, Force: {round(pf, 5)}")
    # print(point_forces_location, point_forces)

    # x_train = [-120, 52, 228, 392, 568, 732, 908, 1080]


    # SFD
    print("\n****Shear Force Diagram****")
    shear_force = []

    for i in range(len(point_forces)):
        shear_force.append(sum(point_forces[:i+1]))

    for sl, sf in zip(point_forces_location, shear_force):
        print(f"Location: {sl}, Shear Force: {round(sf, 5)}")

    for i in range(len(shear_force)):
        
        if i > 0 and i < len(shear_force)-1:
            axes[0].plot([point_forces_location[i], point_forces_location[i]], [shear_force[i-1], shear_force[i]], 'b-')
            axes[0].plot([point_forces_location[i-1], point_forces_location[i]], [shear_force[i-1], shear_force[i-1]], 'b-')
        elif i == 0:
            axes[0].plot([point_forces_location[i], point_forces_location[i]], [0, shear_force[i]], 'b-')
        elif i == len(shear_force)-1:
            axes[0].plot([point_forces_location[i], point_forces_location[i]], [shear_force[i-1], 0], 'b-')
            axes[0].plot([point_forces_location[i-1], point_forces_location[i]], [shear_force[i-1], shear_force[i-1]], 'b-')

    # plt.show()


    # BMD
    print("\n****Bending Moment Diagram****")
    bending_moment = [0]
    accumulated_moment = 0

    for i in range(len(shear_force)-1):
        accumulated_moment += shear_force[i] * (point_forces_location[i+1] - point_forces_location[i] if i < len(shear_force)-1 else point_forces_location[i] - 0)
        bending_moment.append(accumulated_moment)

    for bl, bm in zip(point_forces_location, bending_moment):
        print(f"Location: {bl}, Bending Moment: {round(bm, 5)}")
    # print(bending_moment)
    axes[1].plot(point_forces_location, bending_moment, 'r-')
    plt.show()


def all_bmd(train_1_weight, train_2_weight, train_3_weight, range_start=0, range_end=1200):
    fig, axes = plt.subplots(1, 1, figsize=(10, 4))
    
    train_weight = {
    1: train_1_weight,
    2: train_2_weight,
    3: train_3_weight
    }

    # SET UP
    n_trains = 3
    
    max_moment = -10000000
    first_train_position = -100000
    position_of_max = -100000

    for train_1_location in range(range_start, range_end+1):

        front_train_spacing = 52
        wheel_spacing = 176
        train_spacing = 164

        point_load_pair = {}

        for num in range(n_trains):
            wheel_1_location = train_1_location - num*train_spacing - num*wheel_spacing
            wheel_1_weight = train_weight[num+1]/2

            wheel_2_location = wheel_1_location - wheel_spacing
            wheel_2_weight = train_weight[num+1]/2

            if wheel_1_location >= 0 and wheel_1_location <= 1200:
                point_load_pair[wheel_1_location] = wheel_1_weight
            if wheel_2_location >= 0 and wheel_2_location <= 1200:
                point_load_pair[wheel_2_location] = wheel_2_weight

        total_load = sum(point_load_pair.values())
        # print(point_load_pair)


        total_negative_moment_at_0 = 0
        for location, weight in point_load_pair.items():
            total_negative_moment_at_0 -= weight * location

        # print(total_negative_moment_at_0)


        reaction_force_2 = -total_negative_moment_at_0 / 1200
        reaction_force_1 = total_load - reaction_force_2

        point_forces_location = [0]
        point_forces = [reaction_force_1]

        for location in sorted(point_load_pair.keys()):
            point_forces_location.append(location)
            point_forces.append(-point_load_pair[location])

        point_forces_location.append(1200)
        point_forces.append(reaction_force_2)

        # print(point_forces_location, point_forces)

        # x_train = [-120, 52, 228, 392, 568, 732, 908, 1080]


        # SFD
        shear_force = []

        for i in range(len(point_forces)):
            shear_force.append(sum(point_forces[:i+1]))

        # BMD
        # print("****BMD****")
        bending_moment = [0]
        accumulated_moment = 0

        for i in range(len(shear_force)-1):
            accumulated_moment += shear_force[i] * (point_forces_location[i+1] - point_forces_location[i] if i < len(shear_force)-1 else point_forces_location[i] - 0)
            bending_moment.append(accumulated_moment)

        if round(max([abs(x) for x in bending_moment]), 5) > max_moment:
            max_moment = round(max([abs(x) for x in bending_moment]), 5)
            first_train_position = [train_1_location]
            position_of_max = [point_forces_location[[abs(x) for x in bending_moment].index(max([abs(x) for x in bending_moment]))]]

        if round(max([abs(x) for x in bending_moment]), 5) == max_moment and len(first_train_position) > 0 and first_train_position[-1] != train_1_location:
            first_train_position.append(train_1_location)
            position_of_max.append(point_forces_location[[abs(x) for x in bending_moment].index(max([abs(x) for x in bending_moment]))])

        axes.plot(point_forces_location, bending_moment, 'r-')

    print("\n***** SUMMARY BENDING MOMENT *****")
    print("Max Bending Moment:", max_moment)
    for ftp, pom in zip(first_train_position, position_of_max):
        print(f"  First Train Position: {ftp}, Position of Max Moment: {pom}")

    plt.show()


def all_sfd(train_1_weight, train_2_weight, train_3_weight, self_weight=0, range_start=0, range_end=1200):
    fig, axes = plt.subplots(1, 1, figsize=(10, 4))
    
    train_weight = {
    1: train_1_weight,
    2: train_2_weight,
    3: train_3_weight
    }

    # SET UP
    n_trains = 3

    """
                        _   T3    _         _   T2    _         _  T1     _
    __________._________|_________|_________|_________|_________|_________|__________.
    |<- 120 ->|<- 52 ->|<- 176 ->|<- 164 ->|<- 176 ->|<- 164 ->|<- 176 ->|<- 52 ->|<- 120 ->|
    """

    front_train_spacing = 52
    wheel_spacing = 176
    train_spacing = 164

    max_shear_force = -1000000000000
    first_train_position = -1000000000000000000000000
    position_of_max = []

    for train_1_location in range(range_start, range_end):
        point_load_pair = {}

        for num in range(n_trains):
            wheel_1_location = train_1_location - num*train_spacing - num*wheel_spacing
            wheel_1_weight = train_weight[num+1]/2

            wheel_2_location = wheel_1_location - wheel_spacing
            wheel_2_weight = train_weight[num+1]/2

            if wheel_1_location >= 0 and wheel_1_location <= 1200:
                point_load_pair[wheel_1_location] = wheel_1_weight
            if wheel_2_location >= 0 and wheel_2_location <= 1200:
                point_load_pair[wheel_2_location] = wheel_2_weight

        total_load = sum(point_load_pair.values()) + self_weight
        # print(point_load_pair)


        total_negative_moment_at_0 = 0
        for location, weight in point_load_pair.items():
            total_negative_moment_at_0 -= weight * location

        # print(total_negative_moment_at_0)


        reaction_force_2 = -total_negative_moment_at_0 / 1200
        reaction_force_1 = total_load - reaction_force_2

        point_forces_location = [0]
        point_forces = [reaction_force_1]

        for location in sorted(point_load_pair.keys()):
            point_forces_location.append(location)
            point_forces.append(-point_load_pair[location])

        point_forces_location.append(1200)
        point_forces.append(reaction_force_2)

        # print(point_forces_location, point_forces)

        # x_train = [-120, 52, 228, 392, 568, 732, 908, 1080]


        # SFD
        shear_force = []

        for i in range(len(point_forces)):
            shear_force.append(sum(point_forces[:i+1]))

        if (round(max([abs(x) for x in shear_force]), 5)) > max_shear_force:
            max_shear_force = round(max([abs(x) for x in shear_force]), 5)
            first_train_position = [train_1_location]
            position_of_max = [point_forces_location[[abs(x) for x in shear_force].index(max([abs(x) for x in shear_force]))]]
        
        if round(max([abs(x) for x in shear_force]), 5) == max_shear_force and len(first_train_position) > 0 and first_train_position[-1] != train_1_location:
            first_train_position.append(train_1_location)
            position_of_max.append(point_forces_location[[abs(x) for x in shear_force].index(max([abs(x) for x in shear_force]))])

        # print(shear_force)
        # print(point_forces_location)

        for i in range(len(shear_force)):
            
            if i > 0 and i < len(shear_force)-1:
                axes.plot([point_forces_location[i], point_forces_location[i]], [shear_force[i-1], shear_force[i]], 'b-')
                axes.plot([point_forces_location[i-1], point_forces_location[i]], [shear_force[i-1], shear_force[i-1]], 'b-')
            elif i == 0:
                axes.plot([point_forces_location[i], point_forces_location[i]], [0, shear_force[i]], 'b-')
            elif i == len(shear_force)-1:
                axes.plot([point_forces_location[i], point_forces_location[i]], [shear_force[i-1], 0], 'b-')
                axes.plot([point_forces_location[i-1], point_forces_location[i]], [shear_force[i-1], shear_force[i-1]], 'b-')
   
    print("\n***** SUMMARY SHEAR FORCE *****")
    print("Max Shear Force:", max_shear_force)
    for ftp, position in zip(first_train_position, position_of_max):
        print(f"  First Train Position: {ftp}, Position of Max Shear Force: {position}")
    
    plt.show()

def centroidal_axis_and_moi(self):
    centroidal_axis = 0
    area_total = 0

    for section in self.cross_section_dimensions:
        height = section[0]
        width_top = section[1]
        thickness = section[2]

        area = width_top * thickness
        centroidal_axis += area * (thickness / 2 + height)
        # print("Area: ", area, " Centroidal Axis at: ", (thickness / 2 + height))
        area_total += area

    centroidal_axis /= area_total
    # print("Centroidal Axis from Bottom (mm):", centroidal_axis)

    moment_of_inertia = 0

    for section in self.cross_section_dimensions:
        height = section[0]
        width_top = section[1]
        thickness = section[2]

        area = width_top * thickness
        distance_to_centroid = abs((thickness / 2 + height) - centroidal_axis)
        moment_of_inertia += (width_top * thickness**3) / 12 + area * distance_to_centroid**2

    # print("Moment of Inertia (mm^4):", moment_of_inertia/10**6)
    return centroidal_axis, moment_of_inertia/10**6

def q_at_centroid(self):
    H = 0
    for y_bottom, w, t in self.cross_section_dimensions:
        H = max(H, y_bottom + t)

    sorted_dimensions = sorted(self.cross_section_dimensions, key=lambda x: x[0])

    Q_top = 0.0

    for y_bottom, w, t in sorted_dimensions:
        y_top = y_bottom + t
        if y_top > self.y_bar:
            h_prime = y_top - max(y_bottom, self.y_bar)
            if h_prime > 0:
                A_prime = w * h_prime
                y_c_prime = y_top - (h_prime / 2.0)
                y_bar_prime = abs(y_c_prime - self.y_bar)
                
                Q_top += A_prime * y_bar_prime
                

    Q_bottom = 0.0

    for y_bottom, w, t in sorted_dimensions:
        y_top = y_bottom + t
        
        if y_bottom < self.y_bar:
            h_prime = min(y_top, self.y_bar) - y_bottom
            if h_prime > 0:
                A_prime = w * h_prime
                y_c_prime = y_bottom + (h_prime / 2.0)
                y_bar_prime = abs(y_c_prime - self.y_bar)
                
                Q_bottom += A_prime * y_bar_prime
    
    return Q_top, Q_bottom
