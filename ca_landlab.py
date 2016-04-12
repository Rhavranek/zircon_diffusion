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
which simulates the diffusion of suspended, neutrally buoyant particles in a
turbulent fluid.


"""

import time
import matplotlib
from numpy import where, arange
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
    xn_list.append( Transition((0,2,0), (2,0,0), 1., 'left/down helium motion') )
    xn_list.append( Transition((2,0,0), (0,2,0), 1., 'right/up helium motion') )
    xn_list.append( Transition((1,0,0), (0,2,0), 0.01, 'decay') )
    xn_list.append( Transition((0,1,0), (2,0,0), 0.01, 'decay') )
    xn_list.append( Transition((1,1,0), (0,2,0), 0.01, 'decay') )
    xn_list.append( Transition((1,1,0), (2,0,0), 0.01, 'decay') )
    
    return xn_list
    
    
def main():
    
    # INITIALIZE

    # User-defined parameters
    nr = 100  # number of rows in grid
    nc = 64  # number of columns in grid
    plot_interval = 0.5   # time interval for plotting, sec
    run_duration = 100.0   # duration of run, sec
    report_interval = 10.0  # report interval, in real-time seconds
    
    # Remember the clock time, and calculate when we next want to report
    # progress.
    current_real_time = time.time()
    next_report = current_real_time + report_interval

    # Create grid
    mg = RasterModelGrid(nr, nc, 1.0)
    
    # Make the boundaries be walls
    #mg.set_closed_boundaries_at_grid_edges(True, True, True, True)
    
    # Set up the states and pair transitions.
    ns_dict = { 0 : 'empty', 
                1 : 'filled lattice',
                2 : 'helium' }
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
    
    # For visual display purposes, set all boundary nodes to fluid
    node_state_grid[mg.closed_boundary_nodes] = 0
    
    # Create the CA model
    ca = RasterCTS(mg, ns_dict, xn_list, node_state_grid)
    
    filled = '#696969'
    empty = '#fffff0'
    helium = '#00bfff'
    clist = [empty, filled, helium]
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
        

    # FINALIZE

    # Plot
    ca_plotter.finalize()

# If user runs this file, activate the main() function
if __name__ == "__main__":
    main()