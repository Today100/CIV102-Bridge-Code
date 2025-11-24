from designs.other.tool import BridgeProject

weight = [182, 135, 135]
# weight = [400/3, 400/3, 400/3]
cross_sections = [
    [2.54, 77.46, 11.27, 97.46, 1],
    [3.81, 10, 0, 96.19, 2],
    [3.81, 10, 90, 96.19, 2],
    [96.19, 1.27, 10, 1.27, 3],
    [96.19, 1.27, 88.73, 1.27, 3],
    [1.27, 80, 10, 0, 0],
    [2.54, 1.27, 10, 97.46, 0],
    [2.54, 1.27, 88.73, 97.46, 0]
]

bp = BridgeProject(weight, cross_sections)
# bp.sfd_envelope()

bp.print_cross_section_properties()
bp.draw_cross_section()
glue_tab_position = 97.46
glue_tab_width = 20

diaphram_distance = 200
diaphram_height = 96.19
bp.factor_of_safety(bp.sfd_max, bp.bmd_max, glue_tab_position, glue_tab_width, diaphram_distance, diaphram_height, 1.27) 