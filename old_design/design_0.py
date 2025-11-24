from tool import BridgeProject

# Design 1: Flaps that's so long

weight = [400/3, 400/3, 400/3]
# load_case_2 = 138
# weight = [load_case_2, load_case_2, load_case_2*1.38]
cross_sections = [
        # [height, width, x_left, y_bottom]  (bottom = 0)
        [1.27, 10, 0, 75, 2],
        # [1.27, 6.27, 10, 75, 0],
        [1.27, 80, 10, 75, 1],
        # [1.27, 6.27, 83.73, 75, 0],
        [1.27, 10, 90, 75, 2],
        [73.73, 1.27, 10, 1.27, 3],
        [1.27, 5, 11.27, 73.73, 0],
        [1.27, 5, 83.73, 73.73, 0],
        [73.73, 1.27, 88.73, 1.27, 3],
        [1.27, 80, 10, 0, 0]
    ]

bp = BridgeProject(weight, cross_sections)

bp.print_cross_section_properties()
bp.draw_cross_section()
glue_tab_position = 75
glue_tab_width = 10

diaphram_distance = 400
diaphram_height = 76.27
bp.factor_of_safety(bp.sfd_max, bp.bmd_max, glue_tab_position, glue_tab_width, diaphram_distance, diaphram_height, 1.27) 
bp.plot_buckling_sections()