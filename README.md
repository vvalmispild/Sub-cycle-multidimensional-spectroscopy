# Sub-cycle-multidimensional-spectroscopy
Sub-cycle multidimensional spectroscopy of  strongly correlated materials

Algorithm of multidimensional spectra construction:
1) calculate time dependent Green's functions. Non-equilibrium perturbation is an external electric field(vector potential). Multidimensional spectra require plenty calculation with changing phase(phi) of external electric field. G(t,t',phi)
2) Fourier transform to get time and frequency dependence of all Green's functions (Spectral functions). G(t,t',phi)->G(t,w,phi)
3) Extract the data corresponding energies of Habbard bands and frequency=0 from all Spectral functions.  G(t,w=LHB,phi), G(t,w=0,phi), G(t,w=UHB,phi)
4) Construct 2D CEP_{LHB}=G(t,w=LHB,tau_{phi}), CEP_{w=0}=G(t,w=0,tau_{phi}),CEP_{UHB}=G(t,w=UHB,tau_{phi}) spectra. 
5) Make reconstruction algorithm to extract only non-liniar part of the CEP.
6) Make 2D Fourier transform of reconstructed CEP specta. G(t,w=x,tau_{phi}) -> G(w_{t},w=x,w_{CEP})

steps_ft.py - code for steps 4-6.

algorithm_reconstruction.pdf - file with description of step 5. (what should be done for nonlinear part of CEP extraction).

CEP_14.pdf - file with figures for each steps of reconstruction algorithm and 2D Fourier transform.

CEP_14_DFT_2periods.pdf - Collection of the of the G(w_{t},w=x,w_{CEP}) and G(w_{t},w=x,tau_{phi}) spectra for the different field intensity.

Gabor.py - Gabor transform.

GT_CEP_clean.pdf - file with figures of Gabor transform spectra for the different field intensity.
