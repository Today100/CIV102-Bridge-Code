from designs.other.tool import BridgeProject

weight = [400/3, 400/3, 400/3]
cross_sections = [
    [1.27, 97.46, 1.27, 98.73, 1],
    [97.46, 1.27, 0, 1.27, 3],
    [97.46, 1.27, 98.73, 1.27, 3],
    [1.27, 10, 1.27, 97.46, 0],
    [1.27, 10, 88.73, 97.46, 0],
    [1.27, 100, 0, 0, 0],
    [1.27, 1.27, 0, 98.73, 0],
    [1.27, 1.27, 98.73, 98.73, 0]
]

bp = BridgeProject(weight, cross_sections)

bp.print_cross_section_properties()
bp.draw_cross_section()
bp.factor_of_safety(bp.sfd_max, bp.bmd_max, 98.73, 20, 200, 100, 1.27)