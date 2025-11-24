import matplotlib.pyplot as plt

class BridgePloting:
    def __init__(self, train_weights, cross_section_dimensions, train_1_location=1028):
        self.train_weight = {
            1: train_weights[0],
            2: train_weights[1],
            3: train_weights[2]
            }
        self.cross_section_dimensions = cross_section_dimensions
        # print("where")
        self.n_trains = 3
        self.front_train_spacing = 52
        self.wheel_spacing = 176
        self.train_spacing = 164

        self.train_1_location = train_1_location
        self.point_load_location, self.point_forces = self.point_loads(self.train_1_location)
        self.sfd_at_location = self.sfd_specific(self.point_forces, self.point_load_location)
        self.bmd_at_location = self.bmd_specific(self.sfd_at_location, self.point_load_location)

        self.sfd_max, self.sfd_max_locations = self.sfd_envelope(0, 2057, plot=False)
        # print("there")
        self.bmd_max, self.bmd_max_locations = self.bmd_envelope(0, 2057, plot=False)
        # print("HEre")

        self.y_bar, self.moi = self.centroidal_axis_and_moi()
        self.Q_top, self.Q_bottom = self.q_at_centroid()

    def point_loads(self, location):
        """Calculate point loads and reactions for given train location."""
        point_load_pair = {}

        for num in range(self.n_trains):
            wheel_1_location = location - num*self.train_spacing - num*self.wheel_spacing
            wheel_1_weight = self.train_weight[num+1]/2

            wheel_2_location = wheel_1_location - self.wheel_spacing
            wheel_2_weight = self.train_weight[num+1]/2

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

        # print("****Point Force Locations and Forces****")
        # for pl, pf in zip(point_forces_location, point_forces):
        #     print(f"Location: {pl}, Force: {round(pf, 5)}")

        return point_forces_location, point_forces
    
    def sfd_specific(self, point_forces, point_load_location=None):
        shear_force = []

        for i in range(len(point_forces)):
            shear_force.append(sum(point_forces[:i+1]))
        
        # for sl, sf in zip(self.point_load_location, self.sfd_at_location):
        #     print(f"Location: {sl}, Shear Force: {round(sf, 5)}")

        return shear_force

    def bmd_specific(self, shear_force, point_load_location):
        bending_moment = [0]
        accumulated_moment = 0

        for i in range(len(shear_force)-1):
            accumulated_moment += shear_force[i] * (point_load_location[i+1] - point_load_location[i] if i < len(shear_force)-1 else point_load_location[i] - 0)
            bending_moment.append(accumulated_moment)

        return bending_moment

    def sfd_envelope(self, range_start, range_end, plot=True):
        plt.cla()
        max_shear_force = -1000000000000
        first_train_position = -1000000000000000000000000
        position_of_max = []

        max_shear_forces = []
        for train_1_location in range(range_start, range_end+1):
            pf_location, pf_forces = self.point_loads(train_1_location)
            shear_force = self.sfd_specific(pf_forces, pf_location)
            for i in range(len(shear_force)):
                if i > 0 and i < len(shear_force)-1:
                    plt.plot([pf_location[i], pf_location[i]], [shear_force[i-1], shear_force[i]], 'b-')
                    plt.plot([pf_location[i-1], pf_location[i]], [shear_force[i-1], shear_force[i-1]], 'b-')
                elif i == 0:
                    plt.plot([pf_location[i], pf_location[i]], [0, shear_force[i]], 'b-')
                elif i == len(shear_force)-1:
                    plt.plot([pf_location[i], pf_location[i]], [shear_force[i-1], 0], 'b-')
                    plt.plot([pf_location[i-1], pf_location[i]], [shear_force[i-1], shear_force[i-1]], 'b-')
   

            max_shear_forces.append([train_1_location, max(shear_force), pf_location[shear_force.index(max(shear_force))]])
            if (round(max([abs(x) for x in shear_force]), 5)) > max_shear_force:
                max_shear_force = round(max([abs(x) for x in shear_force]), 5)
                first_train_position = [train_1_location]
                position_of_max = [pf_location[[abs(x) for x in shear_force].index(max([abs(x) for x in shear_force]))]]
            if round(max([abs(x) for x in shear_force]), 5) == max_shear_force and len(first_train_position) > 0 and first_train_position[-1] != train_1_location:
                first_train_position.append(train_1_location)
                position_of_max.append(pf_location[[abs(x) for x in shear_force].index(max([abs(x) for x in shear_force]))])
        print("\n***** SUMMARY SHEAR FORCE *****")
        print("Max Shear Force:", max_shear_force)
        for ftp, pom in zip(first_train_position, position_of_max):
            print(f"  First Train Position: {ftp}, Position of Max Shear Force: {pom}")
       
        if plot:
            plt.show()
        return max_shear_force, zip(first_train_position, position_of_max)

    def bmd_envelope(self, range_start, range_end, plot=True):
        plt.cla()
        max_moment = -10000000
        first_train_position = -100000
        position_of_max = -100000

        moments = []
        for train_1_location in range(range_start, range_end+1):
            pf_location, pf_forces = self.point_loads(train_1_location)
            shear_force = self.sfd_specific(pf_forces, pf_location)
            bending_moment = self.bmd_specific(shear_force, pf_location)

            moments.append([train_1_location, max(bending_moment), pf_location[bending_moment.index(max(bending_moment))]])
           
            plt.plot(pf_location, bending_moment, 'r-')
            if round(max([abs(x) for x in bending_moment]), 5) > max_moment:
                max_moment = round(max([abs(x) for x in bending_moment]), 5)
                first_train_position = [train_1_location]
                position_of_max = [pf_location[[abs(x) for x in bending_moment].index(max([abs(x) for x in bending_moment]))]]
            if round(max([abs(x) for x in bending_moment]), 5) == max_moment and len(first_train_position) > 0 and first_train_position[-1] != train_1_location:
                first_train_position.append(train_1_location)
                position_of_max.append(pf_location[[abs(x) for x in bending_moment].index(max([abs(x) for x in bending_moment]))])

        print("\n***** SUMMARY BENDING MOMENT *****")
        print("Max Bending Moment:", max_moment)
        for ftp, pom in zip(first_train_position, position_of_max):
            print(f"  First Train Position: {ftp}, Position of Max Moment: {pom}")  
        
        if plot:
            plt.show()
        return max_moment, zip(first_train_position, position_of_max)

    def plot_sfd_bmd_specific(self, location=None, plot=True):
        # print("\n****Shear Force Diagram****")
        plt.cla()
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
       
        # for sl, sf in zip(self.point_load_location, self.sfd_at_location):
        #     print(f"Location: {sl}, Shear Force: {round(sf, 5)}")

        if location is not None:
            self.point_load_location, self.point_forces = self.point_loads(location)
            self.sfd_at_location = self.sfd_specific(self.point_forces, self.point_load_location)
            self.bmd_at_location = self.bmd_specific(self.sfd_at_location, self.point_load_location)

        for i in range(len(self.sfd_at_location)):
            
            if i > 0 and i < len(self.sfd_at_location)-1:
                axes[0].plot([self.point_load_location[i], self.point_load_location[i]], [self.sfd_at_location[i-1], self.sfd_at_location[i]], 'b-')
                axes[0].plot([self.point_load_location[i-1], self.point_load_location[i]], [self.sfd_at_location[i-1], self.sfd_at_location[i-1]], 'b-')
            elif i == 0:
                axes[0].plot([self.point_load_location[i], self.point_load_location[i]], [0, self.sfd_at_location[i]], 'b-')
            elif i == len(self.sfd_at_location)-1:
                axes[0].plot([self.point_load_location[i], self.point_load_location[i]], [self.sfd_at_location[i-1], 0], 'b-')
                axes[0].plot([self.point_load_location[i-1], self.point_load_location[i]], [self.sfd_at_location[i-1], self.sfd_at_location[i-1]], 'b-')

        # BMD
        print("\n****Shear Force Diagram****")
        for sl, sf in zip(self.point_load_location, self.sfd_at_location):
            print(f"Location: {sl}, Shear Force: {round(sf, 5)}")
        axes[0].set_title("Shear Force Diagram")
        axes[1].set_title("Bending Moment Diagram")
        print("\n****Bending Moment Diagram****")

        for bl, bm in zip(self.point_load_location, self.bmd_at_location):
            print(f"Location: {bl}, Bending Moment: {round(bm, 5)}")
        # print(bending_moment)
        axes[1].plot(self.point_load_location, self.bmd_at_location, 'r-')
        if plot:
            plt.show()
    
    def plot_sfd_bmd_envelope(self):
        pass

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

    def print_cross_section_properties(self):
        print("\n****CROSS SECTION PROPERTIES****")
        print("Centroidal Axis from Bottom (mm):", self.y_bar)
        print("Moment of Inertia (mm^4):", self.moi)
        print("Q Top (mm^3):", self.Q_top)
        print("Q Bottom (mm^3):", self.Q_bottom)

        print("\n********Shear Force Maximums********")
        print("Max Shear Force:", self.sfd_max)
        for ftp, pom in self.sfd_max_locations:
            print(f"  First Train Position: {ftp}, Position of Max Shear Force: {pom}")

        print("\n********Bending Moment Maximums********")
        print("Max Bending Moment:", self.bmd_max)
        for ftp, pom in self.bmd_max_locations:
            print(f"  First Train Position: {ftp}, Position of Max Moment: {pom}")

weight = [135*1.35, 135, 135]

cross_section_dimension = [
        #yi, Width Top, Thickness
        [0, 80, 1.27],
        [1.27, 1.27, 73.73],
        [1.27, 1.27, 73.73],
        [73.73, 5, 1.27],
        [73.73, 5, 1.27],
        [75, 100, 1.27]

    ]

bp = BridgePloting(weight, cross_section_dimension, 856)

bp.plot_sfd_bmd_specific(856)