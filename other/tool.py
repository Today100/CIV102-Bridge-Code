import matplotlib.pyplot as plt
import numpy as np
import math

class BridgeProject:
    """
    BridgeProject with bottom-based coordinates.

    cross_section_dimensions: list of rectangles in format:
        [height, width, x_left, y_bottom]
    where y_bottom is distance from the BOTTOM of the section (y increases upward).
    Units: mm for geometry. Loads in same units you pass (N or kN).
    """

    def __init__(self, train_weights, cross_section_dimensions, train_1_location=1028):
        # Loads
        self.train_weight = {1: train_weights[0], 2: train_weights[1], 3: train_weights[2]}
        # Cross-section: list of [h, w, x_left, y_bottom]
        self.cross_section_dimensions = cross_section_dimensions

        # Train geometry (same defaults you had)
        self.n_trains = 3
        self.front_train_spacing = 52
        self.wheel_spacing = 176
        self.train_spacing = 164

        # initial train position
        self.train_1_location = train_1_location

        # Compute cross-section properties first
        self.x_bar, self.y_bar = self.compute_centroid()   # centroid coordinates (x from left, y from bottom)
        self.I = self.centroidal_axis_and_moi()            # moment of inertia about horizontal centroidal axis (mm^4)
        self.Q_top, self.Q_bottom = self.q_at_centroid()   # Q above and below centroid (mm^3)

        # Loads & diagrams at initial position
        self.point_load_location, self.point_forces = self.point_loads(self.train_1_location)
        self.sfd_at_location = self.sfd_specific(self.point_forces)
        self.bmd_at_location = self.bmd_specific(self.point_forces, self.point_load_location)

        # Envelopes (sweep). These are potentially slow loops but kept for compatibility.
        self.sfd_max, self.sfd_max_locations = self.sfd_envelope(0, 2057, plot=False)
        self.bmd_max, self.bmd_max_locations = self.bmd_envelope(0, 2057, plot=False)


    # -----------------------
    # LOADING (unchanged behavior)
    # -----------------------
    def point_loads(self, location):
        point_load_pair = {}
        for num in range(self.n_trains):
            wheel_1_location = location - num * self.train_spacing - num * self.wheel_spacing
            wheel_1_weight = self.train_weight[num + 1] / 2
            wheel_2_location = wheel_1_location - self.wheel_spacing
            wheel_2_weight = self.train_weight[num + 1] / 2

            if 0 <= wheel_1_location <= 1200:
                point_load_pair[wheel_1_location] = wheel_1_weight
            if 0 <= wheel_2_location <= 1200:
                point_load_pair[wheel_2_location] = wheel_2_weight

        total_load = sum(point_load_pair.values())
        total_negative_moment_at_0 = sum([-w * loc for loc, w in point_load_pair.items()])

        reaction_force_2 = -total_negative_moment_at_0 / 1200
        reaction_force_1 = total_load - reaction_force_2

        # if location == 856 or location == 
        # print("**** Reaction Forces ****")
        # print("Reaction Force 1:", reaction_force_1)
        # print("Reaction Froce 2:", reaction_force_2)

        locs = [0] + sorted(point_load_pair.keys()) + [1200]
        forces = [reaction_force_1] + [-point_load_pair[l] for l in sorted(point_load_pair.keys())] + [reaction_force_2]

        return locs, forces


    # -----------------------
    # SFD / BMD
    # -----------------------
    def sfd_specific(self, point_forces):
        shear = []
        for i in range(len(point_forces)):
            shear.append(sum(point_forces[:i+1]))
        return shear

    def bmd_specific(self, point_forces, locs):
        shear = self.sfd_specific(point_forces)
        M = 0.0
        moments = [0.0]
        for i in range(len(shear) - 1):
            dx = locs[i+1] - locs[i]
            M += shear[i] * dx
            moments.append(M)
        return moments

    def sfd_envelope(self, range_start=0, range_end=2057, plot=True):
        max_shear = -1e18
        positions = []
        for pos in range(range_start, range_end + 1):
            locs, forces = self.point_loads(pos)
            shear = self.sfd_specific(forces)
            if plot:
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


    # -----------------------
    # CROSS-SECTION GEOMETRY (bottom-based)
    # -----------------------
    def compute_centroid(self):
        """Return (x_bar, y_bar) using bottom-based rectangles [h,w,x_left,y_bottom]."""
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
        Compute I about horizontal centroidal axis (y-axis for bending).
        Returns I (mm^4).
        """
        # centroid already computed in __init__ via compute_centroid -> self.y_bar
        # But to be safe, ensure centroid exists:
        if not hasattr(self, 'y_bar'):
            self.x_bar, self.y_bar = self.compute_centroid()

        I_total = 0.0
        for h, w, x_left, y_bottom, buckle_case in self.cross_section_dimensions:
            A = w * h
            y_c = y_bottom + h / 2.0
            I_local = (w * h**3) / 12.0           # about its own centroid horizontal axis
            d = y_c - self.y_bar
            I_total += I_local + A * d**2
        return I_total


    def q_at_centroid(self):
        """
        Compute Q above and Q below the centroid (first moment of area).
        Q_above: first moment of area of material ABOVE neutral axis (towards top).
        Q_below: first moment of area of material BELOW neutral axis (towards bottom).
        """
        y_bar = self.y_bar
        Q_above = 0.0
        Q_below = 0.0

        # sort rectangles by bottom (lowest to highest)
        rects = sorted(self.cross_section_dimensions, key=lambda r: r[3])

        for h, w, x_left, y_bottom, buckle_case in rects:
            y_top = y_bottom + h

            # Portion above neutral axis (y > y_bar)
            if y_top > y_bar:
                # height of portion above NA
                h_above = y_top - max(y_bar, y_bottom)
                if h_above > 0:
                    A_prime = w * h_above
                    y_c_prime = max(y_bar, y_bottom) + h_above / 2.0
                    Q_above += A_prime * abs(y_c_prime - y_bar)

            # Portion below neutral axis (y < y_bar)
            if y_bottom < y_bar:
                h_below = min(y_top, y_bar) - y_bottom
                if h_below > 0:
                    A_prime = w * h_below
                    y_c_prime = y_bottom + h_below / 2.0
                    Q_below += A_prime * abs(y_c_prime - y_bar)

        return Q_above, Q_below


    # -----------------------
    # Q at arbitrary y (first moment of area ABOVE the cut at y)
    # -----------------------
    def get_Q_above(self, y_query):
        """
        Q of area ABOVE horizontal line y = y_query (bottom-based y).
        """
        Q = 0.0
        for h, w, x_left, y_bottom, buckle_case in self.cross_section_dimensions:
            y_top = y_bottom + h

            # rectangle fully below the cut -> no contribution
            if y_top <= y_query:
                continue

            # rectangle fully above the cut
            if y_bottom >= y_query:
                A = w * h
                y_c = y_bottom + h / 2.0
                Q += A * abs(y_c - self.y_bar)
            else:
                # partial: portion above from y_query to y_top
                h_part = y_top - y_query
                if h_part > 0:
                    A = w * h_part
                    y_c_part = y_query + h_part / 2.0
                    Q += A * abs(y_c_part - self.y_bar)
        return Q

    def get_Q_below(self, y_query):
        """First moment of area BELOW horizontal line y = y_query."""
        Q = 0.0
        for h, w, x_left, y_bottom in self.cross_section_dimensions:
            y_top = y_bottom + h

            # rectangle fully above the cut -> no contribution
            if y_bottom >= y_query:
                continue

            # rectangle fully below the cut
            if y_top <= y_query:
                A = w * h
                y_c = y_bottom + h / 2.0
                Q += A * abs(y_c - self.y_bar)
            else:
                # partial: portion below from y_bottom to y_query
                h_part = y_query - y_bottom
                if h_part > 0:
                    A = w * h_part
                    y_c_part = y_bottom + h_part / 2.0
                    Q += A * abs(y_c_part - self.y_bar)
        return Q


    # -----------------------
    # width at y (sum widths of parts present at that y)
    # -----------------------
    def get_width_at_y(self, y_query):
        """
        Returns total width at horizontal line y = y_query (sums widths of rectangles that include that y).
        """
        b = 0.0
        for h, w, x_left, y_bottom, buckle_case in self.cross_section_dimensions:
            y_top = y_bottom + h
            if y_bottom <= y_query <= y_top:
                b += w
        return b


    # -----------------------
    # STRESSES
    # -----------------------
    def bending_stress(self, y, M):
        """
        Bending stress at vertical coordinate y (mm from BOTTOM):
            sigma = M * (y - y_bar) / I
        """
        if self.I == 0:
            return 0.0
        return M * (y - self.y_bar) / self.I

    def shear_stress(self, y, V):
        """
        Shear stress at vertical coordinate y (mm from BOTTOM) using:
            tau = V * Q(y) / (I * b(y))
        where Q(y) is area above the cut at y (towards top).
        """
        b = self.get_width_at_y(y)
        if b == 0 or self.I == 0:
            return 0.0
        Q = self.get_Q_above(y)
        return V * Q / (self.I * b)


    # -----------------------
    # PLOTTING CROSS SECTION
    # -----------------------
    def draw_cross_section(self, show_centroid=True, annotate=True):
        """
        Draw cross section with bottom-based coordinates (y increases upward).
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
        Compute top/bottom normal stresses and shear at centroidal and glue planes.

        Parameters
        ----------
        V : float
            Shear force at the section (same units as loads).
        M : float
            Bending moment at the section (same units * length).
        y_glue : float
            Vertical coordinate (mm from bottom) of the glue plane where tau_glue is evaluated.

        Returns
        -------
        (sigma_top, sigma_bottom, tau_centroid, tau_glue)
        - sigma_top: normal stress at top fiber (N/mm^2)
        - sigma_bottom: normal stress at bottom fiber (N/mm^2)
        - tau_centroid: shear stress at centroid plane (N/mm^2)
        - tau_glue: shear stress at glue plane (N/mm^2)
        """
        # determine section extents
        if not self.cross_section_dimensions:
            return 0.0, 0.0, 0.0, 0.0

        y_min = min([y_bottom for _, _, _, y_bottom, _ in self.cross_section_dimensions])
        y_max = max([y_bottom + h for h, _, _, y_bottom, _ in self.cross_section_dimensions])

        # top and bottom fiber coordinates (from bottom)
        y_top_fiber = y_max
        y_bottom_fiber = y_min  # usually 0 if bottom flange starts at 0

        # normal stresses: use existing bending_stress helper
        sigma_top = self.bending_stress(y_top_fiber, M)
        sigma_bottom = self.bending_stress(y_bottom_fiber, M)

        # shear at centroid
        b_centroid = self.get_width_at_y(self.y_bar)
        Q_centroid = self.get_Q_above(self.y_bar)
        tau_centroid = 0.0
        if b_centroid != 0 and self.I != 0:
            tau_centroid = V * Q_centroid / (self.I * b_centroid)

        # shear at glue line (user-specified y_glue)
        Q_glue = self.get_Q_above(y_glue)
        print("Q glue is ", Q_glue)
        tau_glue = 0.0
        if b_glue != 0 and self.I != 0:
            tau_glue = V * Q_glue / (self.I * b_glue)

        return sigma_top, sigma_bottom, tau_centroid, tau_glue

    # -----------------------
    # PRINT / UTIL
    # -----------------------
    def print_cross_section_properties(self):
        print("\n****CROSS SECTION PROPERTIES****")
        print(f"Centroid (x̄, ȳ) from bottom: ({self.x_bar:.4f} mm, {self.y_bar:.4f} mm)")
        print(f"Moment of inertia I (mm^4) about horizontal centroidal axis: {self.I:.4f}")
        print(f"Q above centroid (mm^3): {self.Q_top:.4f}")
        print(f"Q below centroid (mm^3): {self.Q_bottom:.4f}")
        print("\n***** SFD/BMD ENVELOPE SUMMARY *****")
        print(f"Max shear (abs): {self.sfd_max:.4f}, positions: {self.sfd_max_locations}")
        print(f"Max moment (abs): {self.bmd_max:.4f}, positions: {self.bmd_max_locations}")
        self.total_area()

    # ------------------------------------------------------------
    # BUCKLING STRESS OF HORIZONTAL PLATES (e.g., top flange)
    # ------------------------------------------------------------
    def buckling_stress_horizontal_plate(self):
        k_table = {
            1: 4.0,
            2: 0.425,
            3: 6,
        }

        results = {1: [], 2: [], 3: []}

        constant = (math.pi**2 * 4000) / (12 * (1 - 0.2**2))  # E=4000 N/mm^2, nu=0.2
        for height, width_top, x_left, y_bottom, buckling_case in self.cross_section_dimensions:
            if buckling_case in k_table:
                if buckling_case == 3:
                    t = width_top
                    b = height+y_bottom - self.y_bar
                    print(buckling_case, b)
                    # + 1.27 if needed
                    # print("Buckling case 3 detected", b)
                else:
                    t = height
                    b = width_top
                k = k_table[buckling_case]
                # print(buckling_case, t, b)
                sigma_cr = (k * constant) * (t / b) ** 2
                results[buckling_case].append(sigma_cr)
        return results


    def plot_buckling_sections(self):
        fig, ax = plt.subplots(figsize=(6,6))
        ax.set_aspect('equal')

        # colors per case
        colors = {
            1: "tab:red",
            2: "tab:blue",
            3: "tab:green"
        }

        xs, ys = [], []

        for h, w, x, y, case in self.cross_section_dimensions:
            if case not in (1, 2, 3):
                continue
            if case == 3:
                h = h-self.y_bar+y
                y = self.y_bar
            # draw plate
            rect = plt.Rectangle(
                (x, y), w, h,
                fill=False,
                linewidth=2,
                edgecolor=colors.get(case, "black"),
                label=f"Case {case}"
            )
            ax.add_patch(rect)

            # label at center
            cx = x + w/2
            cy = y + h/2
            ax.text(cx, cy, f"{round(w, 3)}x{round(h, 3)}", ha='center', va='center', fontsize=8)

            xs.extend([x, x+w])
            ys.extend([y, y+h])

        # bounds
        ax.set_xlim(min(xs)-10, max(xs)+10)
        ax.set_ylim(min(ys)-10, max(ys)+10)

        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        ax.set_title("Buckling Plates")

        # legend (dedupe by case)
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
        k = 5
        constant = (math.pi**2 * 4000) / (12 * (1 - 0.2**2))  # E=4000 N/mm^2, nu=0.2
        sigma_cr = (k * constant) * ((thickness / distance) ** 2 + (thickness / height) ** 2)
        return sigma_cr
        # for h, w, x_left, y_bottom, buckling_case in self.cross_section_dimensions:


    def factor_of_safety(self, V, M, y_glue, b_glue, distance, height, thickness):
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

        # Buckling
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

        labels = list(fos.keys())
        # values = np.array(list(fos.values()), dtype=float)
        raw_values = list(fos.values())
        values = np.array([0 if (v == float("inf") or np.isinf(v)) else v for v in raw_values], dtype=float)


        # find min
        min_key = min(fos, key=fos.get)
        min_value = fos[min_key]
        min_idx = labels.index(min_key)

        # # compute ratio for annotation
        # ratios = values / min_value

        # colors: red for weakest, default for others
        colors = ["tab:red" if i == min_idx else "tab:blue" for i in range(len(values))]

        plt.figure(figsize=(10,6))
        bars = plt.bar(labels, values, color=colors)
        plt.xticks(rotation=40, ha='right')
        plt.ylabel("Factor of Safety")
        plt.title("FOS values (weakest highlighted in red)")

        # annotate each bar with "value (ratio x.xx)"
        for i, bar in enumerate(bars):
            h = bar.get_height()
            label = f"{round(h, 3)}" if not np.isinf(raw_values[i]) else "None"
            plt.text(bar.get_x() + bar.get_width()/2, h + 0.03*max(values), label,
                    ha='center', va='bottom', fontsize=9)

        # add a horizontal line at min_value for reference
        plt.axhline(min_value, color="gray", linestyle="--", linewidth=1)
        plt.text(len(labels)-0.5, min_value, f"  Min FOS = {round(min_value, 3)}", va='center', color='gray')

        plt.ylim(0, max(values)*1.25)
        plt.tight_layout()
        plt.show()

        return fos

    def total_area(self, bridge_length=1260):
        """Compute total cross-sectional area (mm^2)."""
        A_total = 0.0
        for h, w, x_left, y_bottom, buckle_case in self.cross_section_dimensions:
            if w > 5:
                A_total += (w+1) * bridge_length * (h / 1)
                print(f"Adding area for w={w}, h={h}: {w * bridge_length * (h / 1.27)} mm^2")
            else:
                A_total += (h*1) * bridge_length * (w / 1)
                print(f"Adding area for h={h}, w={w}: {h * bridge_length * (w / 1.27)} mm^2")

        print(f"Approximate Area of Matboard Used: {A_total} mm^2")
        print(f"Approximate Area of Matboard Left: {826008 - A_total} mm^2")
        return A_total
    
    def plot_fos_sweep(self,
                   range_start=0,
                   range_end=1200,
                   step=1,
                   y_glue=None,
                   b_glue=1.0,
                   distance=400,
                   height=76.27,
                   thickness=1.27,
                   tensile_strength=30,
                   compressive_strength=6,
                   shear_strength=4,
                   glue_strength=2):
        """
        Sweep train_1_location from range_start to range_end (inclusive, step)
        and compute a single representative FOS (min of modes) for each position.
        Plots min-FOS vs position and marks the global min. Returns (positions, min_fos_list, fos_dicts)
        Arguments for glue/buckling etc mirror existing helpers.
        """
        positions = list(range(range_start, range_end + 1, step))
        min_fos_list = []
        fos_per_pos = []

        # pre-get buckling constants function references
        for pos in positions:
            locs, forces = self.point_loads(pos)
            # shear diagram and moment diagram for that pos
            shear = self.sfd_specific(forces)
            moments = self.bmd_specific(forces, locs)

            # pick representative V and M (use maximum absolute value in the internal diagram)
            V = max([abs(s) for s in shear]) if shear else 0.0
            M = max([abs(m) for m in moments]) if moments else 0.0

            # get applied stresses at user-specified glue plane (default to centroid if not given)
            yg = self.y_bar if y_glue is None else y_glue
            sigma_comp, sigma_ten, tau_centroid, tau_glue = self.applied_stresses(V, M, yg, b_glue)

            # buckling strengths
            sigma_cr_buckle_cases = self.buckling_stress_horizontal_plate()  # returns lists per case
            sigma_cr_shear_buckling = self.shear_buckling(distance, height, thickness)

            # safe divide helper
            def safe_div(n, d):
                try:
                    if d == 0 or d is None:
                        return float('inf')
                    return float(n) / abs(d) if d != 0 else float('inf')
                except:
                    return float('inf')

            fos = {}
            fos["Tensile"] = safe_div(tensile_strength, sigma_ten)
            fos["Compressive"] = safe_div(compressive_strength, sigma_comp)
            fos["Shear"] = safe_div(shear_strength, tau_centroid)
            fos["Glue Shear"] = safe_div(glue_strength, tau_glue)

            # Flexural buckling cases (use first entry per case if present)
            fos["Flex Buck 1 (k=4)"] = (safe_div(sigma_cr_buckle_cases[1][0], sigma_comp)
                                        if len(sigma_cr_buckle_cases.get(1, [])) > 0 else float('inf'))
            fos["Flex Buck 2 (k=0.425)"] = (safe_div(sigma_cr_buckle_cases[2][0], sigma_comp)
                                            if len(sigma_cr_buckle_cases.get(2, [])) > 0 else float('inf'))
            fos["Flex Buck 3 (k=6)"] = (safe_div(sigma_cr_buckle_cases[3][0], sigma_comp)
                                        if len(sigma_cr_buckle_cases.get(3, [])) > 0 else float('inf'))

            fos["Shear Buckling (case shear)"] = safe_div(sigma_cr_shear_buckling, tau_centroid)

            # store and take minimum (ignore inf)
            fos_per_pos.append(fos)
            finite_vals = [v for v in fos.values() if np.isfinite(v)]
            min_fos = min(finite_vals) if finite_vals else float('inf')
            min_fos_list.append(min_fos)

        # Plot results
        plt.figure(figsize=(10,4))
        plt.plot(positions, min_fos_list, '-', linewidth=1.5)
        plt.axhline(1.0, color='gray', linestyle='--', label='FOS = 1.0')
        # mark global minimum
        min_idx = int(np.argmin(min_fos_list))
        plt.plot(positions[min_idx], min_fos_list[min_idx], 'ro', label=f"Min FOS = {min_fos_list[min_idx]:.3f} at x={positions[min_idx]}")
        plt.xlabel("Train 1 position (mm)")
        plt.ylabel("Minimum Factor of Safety (controlling mode)")
        plt.title("FOS sweep along bridge (min of modes at each position)")
        plt.legend()
        plt.grid(alpha=0.25, linestyle='--')
        plt.tight_layout()
        plt.show()

        return positions, min_fos_list, fos_per_pos