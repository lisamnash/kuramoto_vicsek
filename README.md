# Kuramoto-Vicsek Toy Simulation

This repository has some code I wrote as a toy model for the Kuramoto-Vicsek equations.  The integration so far is a simple Euler method.

Details on the Kuramoto-Vicsek model can be found here: https://arxiv.org/abs/1306.3372

I am modeling equations 2.1 and 2.2 from the paper above.  Input parameters can be changed in the file `input.py`.  The output created by this script is an .mp4.  The data is not currently written to a file, but this is will happen in a future version of the code.

## Short Intro to the Kuramoto-Vicsek model:
Flocking is a familiar concept to all of us.  We've all seen flocks of birds, schools of fish, or herds of buffalo.  In all of the cases of flocking in nature, we see long range order emerge from short-range interactions.  Animals (let's call them birds) at opposite ends of the flock cannot communicate, but still seem to move together.  How can this be?

In fact, flocking can be explained in 2D by a very simple model, which has been dubbed the Vicsek model after its originator. The rules for the Vicsek model are simple:
- A large number of point particles (birds) move over time through a space of dimension d.  Each bird attempts to follow its neighbors.  That is to say, that the bird will adjust its orientation to match its neighbors
- The interactions are purely short ranged.  The birds can only see a finite distance around them. 
- Following is not perfect.  This means that each bird tries to match the orientation of its neighbor, but makes mistakes. 
- The underlying model has rotational symmetry: The flock is equally likely to move in any direction. In my model this means that I start the birds with random orientations.

The Kuramoto model describes synchronization of rotating objects in a similar manner.

In this repository you will find the two of these models together.  Our birds will try to align with each other, but they all will want to travel in circles if there is no external alignment.


## Modifying the input parameters

The input parameters of the simulation can be modified in the file `input.py`.  In the version in this repository, I have left spaces for all entries that you might want to modify.  Below is an explanation of each of the input parameters. 

- `initial_flocks` : You may either place birds randomly or place them in flocks.  If you set this parameter each entry in the master list represents a flock.  For example.  In this repository the first entry is [[5., 5.], 1., 50].  This says that the center of one flock is x = 5, y=5.  The flock length is 1.  The number of birds in the flock is 50. 

- `num_points` : Number of birds in simulation.  If no initial flocks are placed then the program will place this number of birds randomly in the starting box.

- `box_size` : The simulation box size.  The simulation has periodic boundary conditions.

- `w_0` : If this or w_amp is non-zero then the birds will try to travel in circles. Birds will try to go in cirlces of radius speed/w. 

-  `w_amp`: Gives amplitude of randomness on circle sizes.  

- `C` : Coefficient for randomness--This represents the mistake the bird makes in alignment.  The higher this number, the more likely it is that birds will make mistakes when trying to align.

- `nu` : Coefficient for alignment.  The higher this coefficient, the quicker a bird will become aligned with its neighbors.  If set to 0, the birds will not align.

- `max_dist` : The maximum distance a bird looks to align.  Represented as faint circle in the movie output.

- `max_num` : Some researchers have actually found that in starling flocks the birds only pay attention to their 7 nearest neighbors despite flock density!  Here you can set the maximum number of birds that each bird will pay attention to (will look at closest `max_num` birds). 

## Running the simulation
To run the simulation, type the following from the directory containing the python scripts :

`python run_sim.py -t $INPUT_TIME -o $OUTPUT_FILE_NAME` 

## Interpreting the output
The output file is an mp4 of your simulation.  The colors of the birds correspond to their orienations.  If they all are the same color then they are aligned.  A light circle around each bird shows its circle of influence.  The bird at the center attempts to align with other birds in its circular neighborhood.