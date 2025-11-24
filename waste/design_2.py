from designs.other.tool import BridgeProject

weight = [400/3, 400/3, 400/3]
cross_sections = [
    [2.54, 77.46, 11.27, 97.46, 1],
    [2.54, 10, 0, 97.46, 2],
    [2.54, 10, 90, 97.46, 2],
    [2.54, 1.27, 10, 97.46, 0],
    [2.54, 1.27, 88.73, 97.46, 0],
    [96.19, 1.27, 10, 1.27, 3],
    [96.19, 1.27, 88.73, 1.27, 3],
    [1.27, 10, 11.27, 96.19, 0],
    [1.27, 10, 78.73, 96.19, 0],
    [1.27, 80, 10, 0, 0]
]

bp = BridgeProject(weight, cross_sections)

bp.print_cross_section_properties()
bp.draw_cross_section()
glue_tab_position = 97.46
glue_tab_width = 20

diaphram_distance = 200
diaphram_height = 97.46
bp.factor_of_safety(bp.sfd_max, bp.bmd_max, glue_tab_position, glue_tab_width, diaphram_distance, diaphram_height, 1.27)