from designs.other.tool import BridgeProject

# Design 7: Small flaps but 66mm bottom and 100mm tall (changed design 3)


weight = [400/3, 400/3, 400/3]
# load_case_2 = 100
# weight = [load_case_2, load_case_2, load_case_2*1.38]
cross_sections = [
    [100, 100, 0, 0, 0]
]

bp = BridgeProject(weight, cross_sections)

# bp.print_cross_section_properties()
# bp.draw_cross_section()
# glue_tab_position = 78
# glue_tab_width = 64

# diaphram_distance = 200
# diaphram_height = 78
# bp.factor_of_safety(bp.sfd_max, bp.bmd_max, glue_tab_position, glue_tab_width, diaphram_distance, diaphram_height, 1.27)
# bp.plot_buckling_sections()

# bp.sfd_envelope()

bp.plot_fos_sweep()