from tool import BridgeProject

# Design 7: Small flaps but 66mm bottom and 100mm tall (changed design 3)


weight = [400/3, 400/3, 400/3]
# load_case_2 = 135
# weight = [load_case_2, load_case_2, load_case_2*1.35]
cross_sections = [
    [2, 54, 23, 98, 1],
    [2, 17, 0, 98, 2],
    [2, 17, 83, 98, 2],
    [2, 6, 17, 98, 0],
    [2, 6, 77, 98, 0],
    [1, 10, 18, 97, 0],
    [1, 10, 72, 97, 0],
    [98, 1, 17, 0, 3],
    [98, 1, 82, 0, 3],
    [1, 64, 18, 0, 0],
    [1, 64, 18, 96, 0],
    # [1, 44, 28, 97, 0]
]

bp = BridgeProject(weight, cross_sections)

bp.print_cross_section_properties()
bp.draw_cross_section()
glue_tab_position = 98
glue_tab_width = 20

diaphram_distance = 100
diaphram_height = 98
bp.factor_of_safety(bp.sfd_max, bp.bmd_max, glue_tab_position, glue_tab_width, diaphram_distance, diaphram_height, 1)
bp.plot_buckling_sections()

# bp.sfd_envelope()

# bp.plot_fos_sweep()