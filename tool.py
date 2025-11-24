import matplotlib.pyplot as plt
import numpy as np
import math

class BridgeProject:
    """
    Code used to generate information and factor of safety based on given cross section dimention and train weights applied.

    train_weights: list of weights of each train in the format of [train 1 weight, train 2 weight, train 3 weight]
        train 1 is defined as the right most train.

    cross_section_dimensions: list of rectangles in format:
        [height, width, x_left, y_bottom, buckle_case]
        where y_bottom is distance from the BOTTOM of the section (y increases upward).
        Units: mm for geometry. Loads in same units you pass (N or kN).
    """

    def __init__(self, train_weights, cross_section_dimensions, train_1_location=1028):
        # store loads per train (simple mapping for readability)
        self.train_weight = {1: train_weights[0], 2: train_weights[1], 3: train_weights[2]}

        # cross-section rectangles: each entry is [h, w, x_left, y_bottom, buckle_case]
        self.cross_section_dimensions = cross_section_dimensions

        # geometry of the train axle/wheel layout (defaults kept)
        self.n_trains = 3
        self.front_train_spacing = 52
        self.wheel_spacing = 176
        self.train_spacing = 164

        # starting location for the frontmost train (mm from left)
        self.train_1_location = train_1_location

        # compute section properties up front so other methods can use them
        self.x_bar, self.y_bar = self.compute_centroid()   # centroid coords
        self.I = self.centroidal_axis_and_moi()            # Moment of Inertia
        self.Q_top, self.Q_bottom = self.q_at_centroid()   # Q above and below  (Compute both to verify accuracy as they should be the same)

        # compute loads and diagrams for the initial position
        self.point_load_location, self.point_forces = self.point_loads(self.train_1_location)
        self.sfd_at_location = self.sfd_specific(self.point_forces)
        self.bmd_at_location = self.bmd_specific(self.point_forces, self.point_load_location)

        # compute envelopes to find maximum sf and bm and related location. 
        self.sfd_max, self.sfd_max_locations = self.sfd_envelope(0, 2057, plot=False)
        self.bmd_max, self.bmd_max_locations = self.bmd_envelope(0, 2057, plot=False)

    def point_loads(self, location):
        """
        Given front train location, return locations and point forces. Assuming the bridge is 1200 m long
        """
        point_load_pair = {}
        for num in range(self.n_trains):
            # compute wheel locations for each train (two wheels per train)
            wheel_1_location = location - num * self.train_spacing - num * self.wheel_spacing
            wheel_1_weight = self.train_weight[num + 1] / 2
            wheel_2_location = wheel_1_location - self.wheel_spacing
            wheel_2_weight = self.train_weight[num + 1] / 2

            # only include wheels that lie on the bridge
            if 0 <= wheel_1_location <= 1200:
                point_load_pair[wheel_1_location] = wheel_1_weight
            if 0 <= wheel_2_location <= 1200:
                point_load_pair[wheel_2_location] = wheel_2_weight

        # total applied load and equivalent negative moment about left support
        total_load = sum(point_load_pair.values())
        total_negative_moment_at_0 = sum([-w * loc for loc, w in point_load_pair.items()])

        # Find support reaction of the support at 1200
        reaction_force_2 = -total_negative_moment_at_0 / 1200
        reaction_force_1 = total_load - reaction_force_2 # Subtracting it from total load will give the other reaction force

        # Save all location and force to a list
        locs = [0] + sorted(point_load_pair.keys()) + [1200]
        forces = [reaction_force_1] + [-point_load_pair[l] for l in sorted(point_load_pair.keys())] + [reaction_force_2]

        return locs, forces


    def sfd_specific(self, point_forces):
        """
        Find shear force based on point forces
        """
        # build shear force list by cumulative sum of point forces (left to right)
        shear = []
        for i in range(len(point_forces)):
            shear.append(sum(point_forces[:i+1]))
        return shear

    def bmd_specific(self, point_forces, locs):
        """
        Find bending moment based on point forces
        """
        # integrate shear piecewise to get bending moment diagram values at each node
        shear = self.sfd_specific(point_forces)
        M = 0.0
        moments = [0.0]
        for i in range(len(shear) - 1):
            dx = locs[i+1] - locs[i]
            M += shear[i] * dx
            moments.append(M)
        return moments

    def sfd_envelope(self, range_start=0, range_end=2057, plot=True):
        # run through front-train positions and record max absolute shear and where it occurs
        max_shear = -1e18
        positions = []
        for pos in range(range_start, range_end + 1):
            locs, forces = self.point_loads(pos)
            shear = self.sfd_specific(forces)
            if plot:
                # quick plotting of the local shear diagram
                for i in range(len(shear)):
                    if i > 0 and i < len(shear)-1:
                        plt.plot([locs[i], locs[i]], [shear[i-1], shear[i]], 'b-')
                        plt.plot([locs[i-1], locs[i]], [shear[i-1], shear[i-1]], 'b-')
                    elif i == 0:
                        plt.plot([locs[i], locs[i]], [0, shear[i]], 'b-')
                    elif i == len(shear)-1:
                        plt.plot([locs[i], locs[i]], [shear[i-1], 0], 'b-')
                        plt.plot([locs[i-1], locs[i]], [shear[i-1], shear[i-1]], 'b-')
            peak = max([abs(s) for s in shear]) if shear else 0
            if peak > max_shear:
                max_shear = peak
                idx = [abs(s) for s in shear].index(max([abs(s) for s in shear]))
                positions = [(pos, locs[idx])]
            elif peak == max_shear:
                idx = [abs(s) for s in shear].index(max([abs(s) for s in shear]))
                positions.append((pos, locs[idx]))
        if plot:
            plt.xlabel("Position along bridge (mm)")
            plt.ylabel("Shear Force (N)")
            plt.title("Shear Force Diagram Envelope")
            plt.grid(True, linestyle='--', alpha=0.25)
            plt.show()
        return max_shear, positions

    def bmd_envelope(self, range_start=0, range_end=2057, plot=True):
        # run through front-train positions and track the maximum absolute moment
        max_M = -1e18
        positions = []
        for pos in range(range_start, range_end + 1):
            locs, forces = self.point_loads(pos)
            M = self.bmd_specific(forces, locs)
            if plot:
                plt.plot(locs, M, 'r-')
            peak = round(max([abs(m) for m in M]), 5) if M else 0
            if peak > max_M:
                max_M = peak
                idx = [abs(m) for m in M].index(max([abs(m) for m in M]))
                positions = [(pos, locs[idx])]
            elif peak == max_M:
                idx = [abs(m) for m in M].index(max([abs(m) for m in M]))
                positions.append((pos, locs[idx]))
        if plot:
            plt.xlabel("Position along bridge (mm)")
            plt.ylabel("Bending Moment (Nmm)")
            plt.title("Bending Moment Envelope")
            plt.grid(True, linestyle='--', alpha=0.25)
            plt.show()
            plt.show()
        return max_M, positions

    def compute_centroid(self):
        """Return (x_bar, y_bar) using inputted cross section geometry [h,w,x_left,y_bottom,...]."""
        A_total = 0.0
        xA = 0.0
        yA = 0.0
        for h, w, x_left, y_bottom, buckle_case, in self.cross_section_dimensions:
            A = w * h
            x_c = x_left + w / 2.0
            y_c = y_bottom + h / 2.0
            A_total += A
            xA += A * x_c
            yA += A * y_c
        if A_total == 0:
            return 0.0, 0.0
        return xA / A_total, yA / A_total

    def centroidal_axis_and_moi(self):
        """
        Calculate moment of inertia
        """
        if not hasattr(self, 'y_bar'):
            self.x_bar, self.y_bar = self.compute_centroid()

        I_total = 0.0
        for h, w, x_left, y_bottom, buckle_case in self.cross_section_dimensions:
            A = w * h
            y_c = y_bottom + h / 2.0
            I_local = (w * h**3) / 12.0           # rectangle about its own centroidal horizontal axis
            d = y_c - self.y_bar                  # parallel axis shift
            I_total += I_local + A * d**2
        return I_total


    def q_at_centroid(self):
        """
        Compute Q above and Q below the centroid (first moment of area).
        Q_above: first moment of area of material ABOVE neutral axis (towards top).
        Q_below: first moment of area of material BELOW neutral axis (towards bottom).
        Q_above === Q_below
        """
        y_bar = self.y_bar
        Q_above = 0.0
        Q_below = 0.0

        # sort rectangles by their bottom y so partial contributions are easier to handle
        rects = sorted(self.cross_section_dimensions, key=lambda r: r[3])

        for h, w, x_left, y_bottom, buckle_case in rects:
            y_top = y_bottom + h

            # handle part above neutral axis
            if y_top > y_bar:
                h_above = y_top - max(y_bar, y_bottom)
                if h_above > 0:
                    A_prime = w * h_above
                    y_c_prime = max(y_bar, y_bottom) + h_above / 2.0
                    Q_above += A_prime * abs(y_c_prime - y_bar)

            # handle part below neutral axis
            if y_bottom < y_bar:
                h_below = min(y_top, y_bar) - y_bottom
                if h_below > 0:
                    A_prime = w * h_below
                    y_c_prime = y_bottom + h_below / 2.0
                    Q_below += A_prime * abs(y_c_prime - y_bar)

        return Q_above, Q_below

    def get_Q_above(self, y_query):
        """
        Q of area ABOVE horizontal line y = y_query (height from bottom).
        Useed for glue shear
        """
        Q = 0.0
        for h, w, x_left, y_bottom, buckle_case in self.cross_section_dimensions:
            y_top = y_bottom + h

            # if rectangle is fully below the cut, skip
            if y_top <= y_query:
                continue

            # rectangle fully above the cut
            if y_bottom >= y_query:
                A = w * h
                y_c = y_bottom + h / 2.0
                Q += A * abs(y_c - self.y_bar)
            else:
                # partial piece above the cut
                h_part = y_top - y_query
                if h_part > 0:
                    A = w * h_part
                    y_c_part = y_query + h_part / 2.0
                    Q += A * abs(y_c_part - self.y_bar)
        return Q

    def get_width_at_y(self, y_query):
        """
        Returns total width at horizontal line y = y_query (adds widths of rectangles that include that y).
        """
        b = 0.0
        for h, w, x_left, y_bottom, buckle_case in self.cross_section_dimensions:
            y_top = y_bottom + h
            if y_bottom <= y_query <= y_top:
                b += w
        return b

    def bending_stress(self, y, M):
        """
        Bending stress at vertical coordinate y (mm from bottom):
            sigma = M * (y - y_bar) / I
        """
        if self.I == 0:
            return 0.0
        return M * (y - self.y_bar) / self.I

    def shear_stress(self, y, V):
        """
        Shear stress at vertical coordinate y (mm from bottom) using:
            tau = V * Q(y) / (I * b(y))
        where Q(y) is area above the cut at y (towards top).
        """
        b = self.get_width_at_y(y)
        if b == 0 or self.I == 0:
            return 0.0
        Q = self.get_Q_above(y)
        return V * Q / (self.I * b)


    def draw_cross_section(self, show_centroid=True, annotate=True):
        """
        Draw cross section with bottom-based coordinates (y increases upward).
        Help to check geometry and centroid location.
        """
        rects = self.cross_section_dimensions
        if not rects:
            print("No cross-section rectangles to draw.")
            return

        x_max = max([x_left + w for _, w, x_left, _, _ in rects])
        y_max = max([y_bottom + h for h, _, _, y_bottom, _ in rects])

        fig, ax = plt.subplots(figsize=(6, max(4, y_max / 50.0)))

        for h, w, x_left, y_bottom, buckle_case in rects:
            ax.add_patch(plt.Rectangle((x_left, y_bottom), w, h, fill=False, linewidth=1.5, color='gray'))
            if annotate:
                ax.text(x_left + w / 2.0, y_bottom + h / 2.0, f"{round(w, 2)}×{round(h, 2)}",
                        ha='center', va='center', fontsize=8, color='red')

        if show_centroid:
            # mark centroidal axis and centroid point for quick verification
            ax.axhline(self.y_bar, color='red', linestyle='--', linewidth=1.2,
                       label=f"Centroidal axis ȳ = {self.y_bar:.2f} mm")
            ax.plot(self.x_bar, self.y_bar, 'ro', label=f"Centroid (x̄={self.x_bar:.1f}, ȳ={self.y_bar:.1f})")

        ax.set_xlim(-20, x_max + 20)
        ax.set_ylim(-20, y_max + 20)   # y increases upward
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel("Width (mm)")
        ax.set_ylabel("Height y (mm) from bottom")
        ax.set_title("Cross-Section (bottom = 0)")
        ax.grid(True, linestyle='--', alpha=0.25)
        ax.legend()
        plt.show()

    def applied_stresses(self, V, M, y_glue, b_glue):
        """
        Compute top/bottom stresses and shear at centroidal and glue planes.
        Returns: (sigma_top, sigma_bottom, tau_centroid, tau_glue)
        """
        if not self.cross_section_dimensions:
            return 0.0, 0.0, 0.0, 0.0

        # find vertical extents of the section
        y_min = min([y_bottom for _, _, _, y_bottom, _ in self.cross_section_dimensions])
        y_max = max([y_bottom + h for h, _, _, y_bottom, _ in self.cross_section_dimensions])

        # top and bottom fiber coordinates (from bottom)
        y_top_fiber = y_max
        y_bottom_fiber = y_min

        # normal stresses via bending formula
        sigma_top = self.bending_stress(y_top_fiber, M)
        sigma_bottom = self.bending_stress(y_bottom_fiber, M)

        # shear at centroid (using Q above centroid)
        b_centroid = self.get_width_at_y(self.y_bar)
        Q_centroid = self.get_Q_above(self.y_bar)
        tau_centroid = 0.0
        if b_centroid != 0 and self.I != 0:
            tau_centroid = V * Q_centroid / (self.I * b_centroid)

        # shear at glue line requested by user
        Q_glue = self.get_Q_above(y_glue)
        print("Q glue is ", Q_glue)
        tau_glue = 0.0
        if b_glue != 0 and self.I != 0:
            tau_glue = V * Q_glue / (self.I * b_glue)

        return sigma_top, sigma_bottom, tau_centroid, tau_glue

    def print_cross_section_properties(self):
        # Summary of section properties and envelope results
        print("\n****CROSS SECTION PROPERTIES****")
        print(f"Centroid (x̄, ȳ) from bottom: ({self.x_bar:.4f} mm, {self.y_bar:.4f} mm)")
        print(f"Moment of inertia I (mm^4) about horizontal centroidal axis: {self.I:.4f}")
        print(f"Q above centroid (mm^3): {self.Q_top:.4f}")
        print(f"Q below centroid (mm^3): {self.Q_bottom:.4f}")
        print("\n***** SFD/BMD ENVELOPE SUMMARY *****")
        print(f"Max shear (abs): {self.sfd_max:.4f}, positions: {self.sfd_max_locations}")
        print(f"Max moment (abs): {self.bmd_max:.4f}, positions: {self.bmd_max_locations}")
        self.total_area()

    def buckling_stress_horizontal_plate(self):
        """
        Compute classical plate buckling stress for plates flagged by their buckling_case.
        k_table maps case -> k (buckling coefficient). Returns dict of lists for each case.
        """
        k_table = {
            1: 4.0,
            2: 0.425,
            3: 6,
        }

        results = {1: [], 2: [], 3: []}

        # constant uses E and nu
        constant = (math.pi**2 * 4000) / (12 * (1 - 0.2**2))  # E=4000 N/mm^2, nu=0.2
        for height, width_top, x_left, y_bottom, buckling_case in self.cross_section_dimensions:
            if buckling_case in k_table:
                if buckling_case == 3:
                    t = width_top
                    b = height+y_bottom - self.y_bar
                    print(buckling_case, b)
                else:
                    t = height
                    b = width_top
                k = k_table[buckling_case]
                sigma_cr = (k * constant) * (t / b) ** 2
                results[buckling_case].append(sigma_cr)
        return results


    def plot_buckling_sections(self):
        # Draw rectangles color-coded by buckling case for visual purpose
        fig, ax = plt.subplots(figsize=(6,6))
        ax.set_aspect('equal')

        colors = {1: "tab:red", 2: "tab:blue", 3: "tab:green"}

        xs, ys = [], []

        for h, w, x, y, case in self.cross_section_dimensions:
            if case not in (1, 2, 3):
                continue
            if case == 3:
                h = h-self.y_bar+y
                y = self.y_bar
            rect = plt.Rectangle(
                (x, y), w, h,
                fill=False,
                linewidth=2,
                edgecolor=colors.get(case, "black"),
                label=f"Case {case}"
            )
            ax.add_patch(rect)

            cx = x + w/2
            cy = y + h/2
            ax.text(cx, cy, f"{round(w, 3)}x{round(h, 3)}", ha='center', va='center', fontsize=8)

            xs.extend([x, x+w])
            ys.extend([y, y+h])

        ax.set_xlim(min(xs)-10, max(xs)+10)
        ax.set_ylim(min(ys)-10, max(ys)+10)

        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        ax.set_title("Buckling Plates")

        # build a small legend based on which cases exist
        handles = []
        seen = set()
        for case in (1, 2, 3):
            if case in [c[-1] for c in self.cross_section_dimensions]:
                if case not in seen:
                    seen.add(case)
                    patch = plt.Rectangle((0,0),1,1,
                                    fill=False,
                                    edgecolor=colors[case],
                                    linewidth=2)
                    handles.append((patch, f"Case {case}"))

        ax.legend([h for h,_ in handles], [l for _,l in handles])

        plt.show()


    def shear_buckling(self, distance, height, thickness):
        # Find shear buckling
        k = 5
        constant = (math.pi**2 * 4000) / (12 * (1 - 0.2**2))  # E=4000 N/mm^2, nu=0.2
        sigma_cr = (k * constant) * ((thickness / distance) ** 2 + (thickness / height) ** 2)
        return sigma_cr


    def factor_of_safety(self, V, M, y_glue, b_glue, distance, height, thickness):
        """
        Compute and print several FOS metrics (tension, compression, shear, glue shear, local buckling).
        Also plots a bar chart highlighting the weakest fos.
        """
        # material strengths (user can override by editing the function)
        tensile_strength = 30  # MPa
        compressive_strength = 6  # MPa
        shear_strength = 4  # MPa
        glue_strength = 2  # MPa

        sigma_comp, sigma_ten, tau_centroid, tau_glue = self.applied_stresses(V, M, y_glue, b_glue)
        print(f"\nApplied stresses:")
        print(f"sigma_comp = {sigma_comp} MPa")
        print(f"sigma_ten = {sigma_ten} MPa")
        print(f"tau_centroid = {tau_centroid} MPa")
        print(f"tau_glue = {tau_glue} MPa")

        # buckling checks
        sigma_cr_buckle_cases = self.buckling_stress_horizontal_plate()
        sigma_cr_shear_buckling = self.shear_buckling(distance, height, thickness)

        print(sigma_cr_buckle_cases)
        for case, stress in sigma_cr_buckle_cases.items():
            print("Local Buckle Case", case, "Stress:", stress)
        print("Shear Buckling Stress:", sigma_cr_shear_buckling)

        print("\n--- FORMULAS WITH NUMBERS ---")

        print(f"Tension FOS = {tensile_strength} / |{sigma_ten}|")
        print(f"Compression FOS = {compressive_strength} / |{sigma_comp}|")
        print(f"Shear FOS = {shear_strength} / {tau_centroid}")
        print(f"Glue shear FOS = {glue_strength} / {tau_glue}")

        print(f"Local Buckling Case 1 FOS = {sigma_cr_buckle_cases[1][0]} / |{sigma_comp}|") if len(sigma_cr_buckle_cases[1]) > 0 else None
        print(f"Local Buckling Case 2 FOS = {sigma_cr_buckle_cases[2][0]} / |{sigma_comp}|") if len(sigma_cr_buckle_cases[2]) > 0 else None
        print(f"Local Buckling Case 3 FOS = {sigma_cr_buckle_cases[3][0]} / |{sigma_comp}|") if len(sigma_cr_buckle_cases[3]) > 0 else None

        print(f"Shear Buckling FOS = {sigma_cr_shear_buckling} / {tau_centroid}")

        # numeric FOS dictionary (guard for zeros/infs)
        fos = {
            "Tensile": tensile_strength / abs(sigma_ten),
            "Compressive": compressive_strength / abs(sigma_comp),
            "Shear": shear_strength / tau_centroid,
            "Glue Shear": glue_strength / tau_glue,
            "Flex Buck 1 (k=4)": sigma_cr_buckle_cases[1][0] / abs(sigma_comp) if len(sigma_cr_buckle_cases[1]) > 0 else float('inf'),
            "Flex Buck 2 (k=0.425)": sigma_cr_buckle_cases[2][0] / abs(sigma_comp) if len(sigma_cr_buckle_cases[2]) > 0 else float('inf'),
            "Flex Buck 3 (k=6)": sigma_cr_buckle_cases[3][0] / abs(sigma_comp) if len(sigma_cr_buckle_cases[3]) > 0 else float('inf'),
            "Shear Buckling (k=5)": sigma_cr_shear_buckling / tau_centroid
        }

        print("\n**** FACTOR OF SAFETY ****")
    
        for fos_type, fos_value in fos.items():
            print(f"{fos_type}: {fos_value}")

        print("")
        min_key = min(fos, key=fos.get)
        min_value = fos[min_key]

        print(f"**** Minimum Factor of Safety: {min_value}")
        print(f"**** Critical Failure Mode: {min_key}")
        print(f"**** Applied Weight: {sum(self.train_weight.values())} N")
        print(f"**** Failure Weight: {sum(self.train_weight.values())* min_value} N / {sum(self.train_weight.values())* min_value/9.81} kg")

        # prepare bar plot where weakest bar is highlighted red
        labels = list(fos.keys())
        raw_values = list(fos.values())
        values = np.array([0 if (v == float("inf") or np.isinf(v)) else v for v in raw_values], dtype=float)

        min_key = min(fos, key=fos.get)
        min_value = fos[min_key]
        min_idx = labels.index(min_key)

        colors = ["tab:red" if i == min_idx else "tab:blue" for i in range(len(values))]

        plt.figure(figsize=(10,6))
        bars = plt.bar(labels, values, color=colors)
        plt.xticks(rotation=40, ha='right')
        plt.ylabel("Factor of Safety")
        plt.title("FOS values (weakest highlighted in red)")

        # annotate each bar with its numeric FOS (or None)
        for i, bar in enumerate(bars):
            h = bar.get_height()
            label = f"{round(h, 3)}" if not np.isinf(raw_values[i]) else "None"
            plt.text(bar.get_x() + bar.get_width()/2, h + 0.03*max(values), label,
                    ha='center', va='bottom', fontsize=9)

        plt.axhline(min_value, color="gray", linestyle="--", linewidth=1)
        plt.text(len(labels)-0.5, min_value, f"  Min FOS = {round(min_value, 3)}", va='center', color='gray')

        plt.ylim(0, max(values)*1.25)
        plt.tight_layout()
        plt.show()

        return fos

    def total_area(self, bridge_length=1260):
        """Compute approximate total cross-sectional area (mm^2) for material estimate."""
        A_total = 0.0
        for h, w, x_left, y_bottom, buckle_case in self.cross_section_dimensions:
            if w > 5:
                # approximate area contribution for wide elements (simple heuristic)
                A_total += (w+1) * bridge_length * (h / 1)
                print(f"Adding area for w={w}, h={h}: {w * bridge_length * (h / 1.27)} mm^2")
            else:
                A_total += (h*1) * bridge_length * (w / 1)
                print(f"Adding area for h={h}, w={w}: {h * bridge_length * (w / 1.27)} mm^2")

        print(f"Approximate Area of Matboard Used: {A_total} mm^2")
        print(f"Approximate Area of Matboard Left: {826008 - A_total} mm^2")
        return A_total
    