from tool import BridgeProject

# Design 5: Big flaps but 66mm bottom and 80mm tall


# weight = [400/3, 400/3, 400/3]
load_case_2 = 300
weight = [load_case_2, load_case_2, load_case_2*1.38]
cross_sections = [
    [2, 66, 17, 78, 0],
    [2, 17, 0, 78, 2],
    [2, 17, 83, 78, 2],
    [1, 32, 18, 77, 0],
    [1, 32, 50, 77, 0],
    [78, 1, 17, 0, 3],
    [78, 1, 82, 0, 3],
    [1, 64, 18, 0, 0]
]

bp = BridgeProject(weight, cross_sections)

bp.print_cross_section_properties()
bp.draw_cross_section()
glue_tab_position = 78
glue_tab_width = 64

diaphram_distance = 200
diaphram_height = 78
bp.factor_of_safety(bp.sfd_max, bp.bmd_max, glue_tab_position, glue_tab_width, diaphram_distance, diaphram_height, 1.27)
bp.plot_buckling_sections()
