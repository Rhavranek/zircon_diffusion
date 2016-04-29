# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 15:44:15 2016

@author: rachelhavranek
#this is a code to simulate radiation damage and helium diffusion in zircon. 
#The blue guys are heliums and the grey boxes are intact lattice pieces - I intend to 
#make this code more complicated for my final project
"""
"""
isotropic_turbulent_suspension.py
Example of a continuous-time, stochastic, pair-based cellular automaton model, 

"""

import time
import matplotlib
import matplotlib.pyplot as plt
from numpy import where, arange, bincount, zeros
from landlab import RasterModelGrid
from landlab.ca.celllab_cts import Transition, CAPlotter
from landlab.ca.raster_cts import RasterCTS


def setup_transition_list():
    """
    Creates and returns a list of Transition() objects to represent state
    transitions for a biased random walk, in which the rate of downward
    motion is greater than the rate in the other three directions.
    
    Parameters
    ----------
    (none)
    
    Returns
    -------
    xn_list : list of Transition objects
        List of objects that encode information about the link-state transitions.
    
    Notes
    -----
    State 0 represents fluid and state 1 represents a particle (such as a 
    sediment grain, tea leaf, or dissolved heavy particle).
    
    The states and transitions are as follows:
    Pair state      Transition to       Process             Rate (cells/s)
    ==========      =============       =======             ==============
    0 (0-0)         (none)              -                   -
    1 (0-1)         2 (1-0)             left/down motion    10.0
    2 (1-0)         1 (0-1)             right/up motion     10.0
    3 (1-1)         (none)              -                   -
    
    """
    
    # Create an empty transition list
    xn_list = []
    
    # Append two transitions to the list.
    # Note that the arguments to the Transition() object constructor are:
    #  - Tuple representing starting pair state
    #    (left/bottom cell, right/top cell, orientation)
    #  - Tuple representing new pair state
    #    (left/bottom cell, right/top cell, orientation)
    #  - Transition rate (cells per time step, in this case 1 sec)
    #  - Name for transition
    xn_list.append( Transition((0,2,0), (2,0,0), 2.0, 'left/down helium motion') )
    xn_list.append( Transition((2,0,0), (0,2,0), 2.0, 'right/up helium motion') )
    xn_list.append( Transition((1,0,0), (3,2,0), 0.000004, 'decayof238' )) #based on ppm in rock - 4 moleculs) )
    xn_list.append( Transition((0,1,0), (2,3,0), 0.000003, 'decayof238') )
    xn_list.append( Transition((1,0,0), (4,2,0), 0.000004, 'decay of Uleft238') )
    xn_list.append( Transition((0,1,0), (2,4,0), 0.000003, 'decay of Uleft238') )
    xn_list.append( Transition((3,0,0), (2,3,0), 0.015, 'movement of u238 right/up'))#based on decay rate of u238) )
    xn_list.append( Transition((3,1,0), (2,3,0), 0.015, 'movement of u238 right/up'))
    xn_list.append( Transition((0,4,0), (4,2,0), 0.015, 'movement of u238 left/down') )
    xn_list.append( Transition((1,4,0), (4,2,0), 0.015, 'movement of u238 left/down') )
    xn_list.append( Transition((2,4,0), (4,2,0), 0.015, 'let a nucleus swap with an alpha') )
    xn_list.append( Transition((4,2,0), (2,4,0), 0.015, 'let a nucleus swap with an alpha') )
    xn_list.append( Transition((2,3,0), (3,2,0), 0.015, 'let a nucleus swap with an alpha') )
    xn_list.append( Transition((3,2,0), (2,3,0), 0.015, 'let a nucleus swap with an alpha') )
    xn_list.append( Transition((3,0,0), (3,0,0), 0.0002, 'stopping of U238right'))#moves 8 times then stops) )
    xn_list.append( Transition((0,3,0), (0,3,0), 0.0002, 'stopping of U238right') )
    xn_list.append( Transition((4,0,0), (3,0,0), 0.0002, 'stopping of U238left') )
    xn_list.append( Transition((0,4,0), (0,3,0), 0.0002, 'stopping of U238left') )
    
    xn_list.append( Transition((1,0,0), (5,2,0), 0.000001, 'decayof235') )
    xn_list.append( Transition((0,1,0), (2,5,0), 0.000001, 'decayof235') )
    xn_list.append( Transition((1,0,0), (6,2,0), 0.000001, 'decay of U235left') )
    xn_list.append( Transition((0,1,0), (2,6,0), 0.000001, 'decay of U235left') )
    xn_list.append( Transition((5,0,0), (2,5,0), 0.098, 'movement of uranium right/up')) #based on decay rate u235) )
    xn_list.append( Transition((5,1,0), (2,5,0), 0.098, 'movement of uranium right/up') )
    xn_list.append( Transition((0,6,0), (6,2,0), 0.098, 'movement of uranium left/down') )
    xn_list.append( Transition((1,6,0), (6,2,0), 0.098, 'movement of uranium left/down') )
    xn_list.append( Transition((2,6,0), (6,2,0), 0.098, 'let a nucleus swap with an alpha') )
    xn_list.append( Transition((6,2,0), (2,6,0), 0.098, 'let a nucleus swap with an alpha') )
    xn_list.append( Transition((2,5,0), (5,2,0), 0.098, 'let a nucleus swap with an alpha') )
    xn_list.append( Transition((5,2,0), (2,5,0), 0.098, 'let a nucleus swap with an alpha') )
    xn_list.append( Transition((5,0,0), (5,0,0), 0.0014, 'stopping of Uright'))#moves 7 times then stops) )
    xn_list.append( Transition((0,5,0), (0,5,0), 0.0014, 'stopping of Uright') )
    xn_list.append( Transition((6,0,0), (6,0,0), 0.0014, 'stopping of Uleft') )
    xn_list.append( Transition((0,6,0), (0,6,0), 0.0014, 'stopping of Uleft') )
    
    
    xn_list.append( Transition((1,0,0), (7,2,0), 0.000001, 'decay') )
    xn_list.append( Transition((0,1,0), (2,7,0), 0.000001, 'decay') )
    xn_list.append( Transition((1,0,0), (8,2,0), 0.000001, 'decay of Uleft') )
    xn_list.append( Transition((0,1,0), (2,8,0), 0.000001, 'decay of Uleft') )
    xn_list.append( Transition((7,0,0), (2,7,0), 0.004, 'movement of uranium right/up'))#based on decay rate of th) )
    xn_list.append( Transition((7,1,0), (2,7,0), 0.004, 'movement of uranium right/up') )
    xn_list.append( Transition((0,8,0), (8,2,0), 0.004, 'movement of uranium left/down') )
    xn_list.append( Transition((1,8,0), (8,2,0), 0.004, 'movement of uranium left/down') )
    xn_list.append( Transition((7,0,0), (7,0,0), 0.00006, 'stopping of Uright'))#moves 6 times then stops) )
    xn_list.append( Transition((2,6,0), (6,2,0), 0.004, 'let a nucleus swap with an alpha') )
    xn_list.append( Transition((6,2,0), (2,6,0), 0.004, 'let a nucleus swap with an alpha') )
    xn_list.append( Transition((2,5,0), (5,2,0), 0.004, 'let a nucleus swap with an alpha') )
    xn_list.append( Transition((5,2,0), (2,5,0), 0.004, 'let a nucleus swap with an alpha') )
    xn_list.append( Transition((0,7,0), (0,7,0), 0.00006, 'stopping of Uright') )
    xn_list.append( Transition((8,0,0), (7,0,0), 0.00006, 'stopping of Uleft') )
    xn_list.append( Transition((0,8,0), (0,7,0), 0.00006, 'stopping of Uleft') )
    return xn_list
    
    
def main():
    
    # INITIALIZE

    # User-defined parameters
    nr = 100  # number of rows in grid
    nc = 64  # number of columns in grid
    plot_interval = 1.0   # time interval for plotting, sec -- each plot is 10^7 years
    run_duration = 100.0   # duration of run, sec 
    report_interval = 10.0  # report interval, in real-time seconds
    
    # Remember the clock time, and calculate when we next want to report
    # progress.
    current_real_time = time.time()
    next_report = current_real_time + report_interval

    # Create grid
    mg = RasterModelGrid(nr, nc, 1.0)
    
    # Make array to record He count through time
    he = zeros(int(run_duration/plot_interval) + 1)
    time_step_count = 1
    
    # Make the boundaries be walls
    #mg.set_closed_boundaries_at_grid_edges(True, True, True, True)
    
    # Set up the states and pair transitions.
    ns_dict = { 0 : 'empty', 
                1 : 'filled lattice',
                2 : 'helium',
                3 : 'uranium238right',
                4 : 'uranium238left',
                5 : 'uranium235right',
                6 : 'uranium235left',
                7 : 'thorium232right',
                8 : 'thorium232left'}
    xn_list = setup_transition_list()

    # Create the node-state array and attach it to the grid
    node_state_grid = mg.add_zeros('node', 'node_state_map', dtype=int)
    
    # Initialize the node-state array: here, the initial condition is a pile of
    # resting grains at the bottom of a container.
    #odd_rows = where(mg.node_y<0.1*nr)[0]
    node_state_grid[:] = arange(nr*nc, dtype=int) % 2
    for r in range(0, nr, 2):
        for c in range(nc):
            n = r*nc + c
            node_state_grid[n] = 1 - node_state_grid[n]
    print(node_state_grid)
    
    node_state_grid[0]=8
    
    # For visual display purposes, set all boundary nodes to fluid
    node_state_grid[mg.closed_boundary_nodes] = 0
    
    # Create the CA model
    ca = RasterCTS(mg, ns_dict, xn_list, node_state_grid)
    
    filled = '#cdcdc1'
    empty = '#ffffff'
    helium = '#7fffd4'
    uranium238right = '#ff1493' #pink
    uranium238left = '#ff1493' #pink
    uranium235right = '#8a2be2' #purple
    uranium235left = '#8a2be2' #purple
    thorium232right = '#f08080' #coral
    thorium232left = '#f08080' #coral 
    clist = [empty, filled, helium, uranium238right, uranium238left, uranium235right, uranium235left, thorium232right, thorium232left]
    my_cmap = matplotlib.colors.ListedColormap(clist)

    # Create a CAPlotter object for handling screen display
    ca_plotter = CAPlotter(ca, cmap=my_cmap)
    
    # Plot the initial grid
    ca_plotter.update_plot()

    # RUN
    current_time = 0.0
    while current_time < run_duration:
        
        # Once in a while, print out simulation and real time to let the user
        # know that the sim is running ok
        current_real_time = time.time()
        if current_real_time >= next_report:
            print 'Current sim time',current_time,'(',100*current_time/run_duration,'%)'
            next_report = current_real_time + report_interval
        
        # Run the model forward in time until the next output step
        ca.run(current_time+plot_interval, ca.node_state, 
               plot_each_transition=True)
        current_time += plot_interval
        
        # Plot the current grid
        ca_plotter.update_plot()
        
        print bincount(node_state_grid)
        he[time_step_count] = bincount(node_state_grid)[2]
        time_step_count += 1
        #plt.plot(bincount(node_state_grid))
        #plt.ylabel('number of particles')

   # FINALIZE

    # Plot

    ca_plotter.finalize()
    
    plt.figure(2)
    plt.plot(he)
    plt.ylabel('Number of Helium Atoms')
    plt.xlabel('Run')
    plt.title('Number of helium particles present in model')
    plt.show()
    

# If user runs this file, activate the main() function
if __name__ == "__main__":
    main()