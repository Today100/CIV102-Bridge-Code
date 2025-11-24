from tool import BridgeProject

# Design 4: Flaps at 20mm

# weight = [400/3, 400/3, 400/3]
load_case_2 = 300
weight = [load_case_2, load_case_2, load_case_2*1.38]
cross_sections = [
    [2, 68, 16, 98, 1],
    [2, 10, 0, 98, 2],
    [2, 10, 90, 98, 2],
    [2, 6, 10, 98, 0],
    [2, 6, 84, 98, 0],
    [1, 20, 11, 97, 0],
    [1, 20, 69, 97, 0],
    # [98, 1, 10, 0, 3],
    [97, 1, 10, 1, 3],
    # [98, 1, 89, 0, 3],
    [97, 1, 89, 1, 3],
    # [1, 78, 11, 0, 0],
    [1, 80, 10, 0, 0]
]

bp = BridgeProject(weight, cross_sections)

bp.print_cross_section_properties()
bp.draw_cross_section()
glue_tab_position = 98
glue_tab_width = 40

diaphram_distance = 200
diaphram_height = 98
bp.factor_of_safety(bp.sfd_max, bp.bmd_max, glue_tab_position, glue_tab_width, diaphram_distance, diaphram_height, 1.27)
bp.plot_buckling_sections()
