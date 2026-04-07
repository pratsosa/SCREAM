## Dev Folder Information

This folder is where I am doing some development work to integrate into SCREAM  
Currently I am working on creating the data downloading and processing pipeline


### fetcher.py
The most up-to-date data downloading file created by my collaborator. 
Contains scripts to crossmatch Gaia with Decals dataset and download it.
We will need to use functions from this file to download the data 

### adql_utils.py
This script was written previously by my collaborator and is less up-to-date than fetcher.py but may contain some useful functions. Is used to download Gaia data using adql. Specifically the functions to transform coordinates may be useful. 

### GD1_CWoLA_Data_Prep_Script.py

This python file was created from the notebook I had previously been using to download + process Gaia data. It crossmatches the data with Streamfinder data I have saved in order to get the Streamfinder labels. It also applies the quality cuts we have decided on. Importantly, it defines the signal + sideband regions. For CATHODE we only need to define the signal region. 


#### Extinction Correction
My collaborator uses the following code/coefficients for extinction correction:

```
t['gmag'], t['rmag'], t['zmag'] = [22.5-2.5*np.log10(t['flux_'+_]) for _ in 'grz']

t['gmag0'] = t['gmag'] - t['ebv'] * 3.186
t['rmag0'] = t['rmag'] - t['ebv'] * 2.140
t['zmag0'] = t['zmag'] - t['ebv'] * 1.196
```