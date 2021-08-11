# EPW-descriptor
The code for Physics-based Feature Makes Machine Learning Cognizing Crystal Physical Properties Simple.
Follow the four steps to get the documentation of the EPW feature:   
(i) Install the python library packages: SeeK-path and Pymatgen, which are dependency libraries for getting the symmetric information.   
(ii) Converting the crystal structure files to POSCAR, which is accepted by Vienna Ab initio Simulation Package (VASP), and we have considered the opportunity for subsequent work to interact with the computational simulation.  
(iii) Place the crystal structure file in the path specified by ‘run.py’ and run ‘run.py’ to generate the ‘EPW.csv’ file.  
(iv) Following the previous three steps, we get the original features of EPW, we recommend do logarithmic processing of the original features before using it for machine learning (ML) work.   
All our work was completed in Python (version 3.7.6) environment.
