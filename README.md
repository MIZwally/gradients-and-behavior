Authors: Mia Zwally, Ka Chun Lam, Adam Thomas, and Dustin Moraczewski.

Preregistration: https://osf.io/t9dhk

This repository contains the code for the project "Characterizing the Relationship Between Cortical Gradients and Cognitive Traits in Children." The ABCD data used in this study requires a data use agreement and therefore cannot be made public.

All of the code for the main analysis is in the root directory. Files that start with '1' are used to preprocess, create, and analyze the gradients, while those that start with '2' are used to preprocess the behavioral data. Both of these sets of scripts need to be run before those that start with '3', which conduct the CCA and require the derivatives of the previous files. The number after the decimal indicates the order in which the scripts should be run. Any scripts that act as packages for the main line are located in the 'Packages' directory.

All paths will need to be changed to match your directory structure.
