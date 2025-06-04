# imports non-AI

# imports AI-specific

# define arguments

# read yaml

# formatting prompt function

# generation function

# running on command line

## instantiate arguments, data, hyperparameters

## load models

## data loop
## provide current clinical note (probably those considered stigmatizing) as "initial response" and request critique using the constitution
    ### during inference will need to chunk the clinical note to locate the area where the langauge can be found

## provide critique to develop revision

## repeat top two steps n times

## use initial and final response for SFT

## provide revisions for DPO

## save model and data