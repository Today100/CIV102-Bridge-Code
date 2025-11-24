from tool import BridgeProject

# Design 1: Flaps that's so long

weight = [400/3, 400/3, 400/3]
# load_case_2 = 135
# weight = [load_case_2, load_case_2, load_case_2*1.35]
cross_sections = [
    [2, 80, 10, 98, 0],
    [2, 10, 0, 98, 2],
    [2, 10, 90, 98, 2],
    [1, 39, 11, 97, 0], 
    [1, 39, 50, 97, 0],
    [97, 1, 10, 1, 3],
    [97, 1, 89, 1, 3],
    [1, 80, 10, 0, 0]
    
]

bp = BridgeProject(weight, cross_sections)

bp.print_cross_section_properties()
bp.draw_cross_section()
glue_tab_position = 98
glue_tab_width = 78

diaphram_distance = 100
diaphram_height = 98
bp.factor_of_safety(bp.sfd_max, bp.bmd_max, glue_tab_position, glue_tab_width, diaphram_distance, diaphram_height, 1) 
bp.plot_buckling_sections()