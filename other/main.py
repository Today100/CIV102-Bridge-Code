import designs.other.calculation as calculation
from designs.other.calculation import BridgeProject
import cross_section





if __name__ == "__main__":
    
#     # INPUTS
    
#     # Adjust weights of each train. Train 1 is the rightmost train.
    train_1_weight = 400/3
    train_2_weight = 400/3
    train_3_weight = 400/3

#     # Adjust train 1 first wheel location. 
#     # Note: Length of train is 856
    train_1_location = 1

    cross_section_dimension = [
        #yi, Width Top, Thickness
        [0, 80, 1.27],
        [1.27, 1.27, 73.73],
        [1.27, 1.27, 73.73],
        [73.73, 5, 1.27],
        [73.73, 5, 1.27],
        [75, 100, 1.27]

    ]

    design_0 = calculation.BridgeProject([train_1_weight, train_2_weight, train_3_weight], cross_section_dimension, train_1_location)
    design_0.sfd_envelope(0, 2056)
    

#     # CROSS SECTION DIMENSIONS
#     # Format: [height from bottom (mm), top width (mm), thickness (mm)]
    # cross_section_dimention = [
    #     #yi, Width Top, Thickness
    #     [0, 80, 1.27],
    #     [1.27, 1.27, 73.73],
    #     [1.27, 1.27, 73.73],
    #     [73.73, 5, 1.27],
    #     [73.73, 5, 1.27],
    #     [75, 100, 1.27]

    # ]
    # y_bar, moi = cross_section.centroidal_axis_and_moi(cross_section_dimention)
    # print(cross_section.q_at_centroid(cross_section_dimention, y_bar))
    # print("Q Top (mm^3):", Q_top/10**3)
    # print("Q Bottom (mm^3):", Q_bottom/10**3)


#     # Plot SFD and BMD for given train 1 location
    # sfd_bmd.plot_sfd_bmd(train_1_weight, train_2_weight, train_3_weight, train_1_location=train_1_location)

#     # Plot all BMD for train 1 locations in given range
    # calculation.all_sfd(train_1_weight, train_2_weight, train_3_weight, range_start=0, range_end=2057)

#     # # Plot all SFD for train 1 locations in given range
    # sfd_bmd.all_sfd(train_1_weight, train_2_weight, train_3_weight, range_start=0, range_end=2057)

