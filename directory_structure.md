<root>:.
|   .gitignore
|   environment.yml
|   fm\_vit.png
|   README.md
|  
+---comparisons
|       **init**.py
|  
+---config
|   |   argument\_parser.py
|   |   data\_config.py
|   |   data\_config\_vis.py
+---data
|       CE-CRP\_mu.npy
|       CE-CRP\_var.npy
|       sample\_PBD-BE1D.png
|       sample\_PBD-CFD1D.png
|       sample\_PBD-CFD2D(IC).png
|       sample\_PBD-CFD2D.png
|       sample\_PBD-CFD3D(Turb).png
|       sample\_PBD-CFD3D.png
|       sample\_PBD-DR1D.png
|       sample\_PBD-DR2D.png
|       sample\_PBD-SW2D.png
|       sample\_PG-CE-CRP2D.png
|       sample\_PG-CE-Gauss2D.png
|       sample\_PG-CE-KH2D.png
|       sample\_PG-CE-RP2D.png
|       sample\_PG-FNS-KF2D.png
|       sample\_PG-NS-Gauss2D.png
|       sample\_PG-NS-Sines2D.png
|       sample\_TW-GSDR2D.png
|       sample\_TW-MHD3D.png
|       sample\_TW-TGC3D.png
|       stats\_be1d\_mu.npy
|       stats\_be1d\_var.npy
|       stats\_ce\_crp\_2d\_mu.npy
|       stats\_ce\_crp\_2d\_var.npy
|       stats\_ce\_gauss\_2d\_mu.npy
|       stats\_ce\_gauss\_2d\_var.npy
|       stats\_ce\_kh\_2d\_mu.npy
|       stats\_ce\_kh\_2d\_var.npy
|       stats\_ce\_rp\_2d\_mu.npy
|       stats\_ce\_rp\_2d\_var.npy
|       stats\_cfd1d\_mu.npy
|       stats\_cfd1d\_var.npy
|       stats\_cfd2d-ic\_mu.npy
|       stats\_cfd2d-ic\_var.npy
|       stats\_cfd2d\_mu.npy
|       stats\_cfd2d\_var.npy
|       stats\_cfd3d-turb\_mu.npy
|       stats\_cfd3d-turb\_var.npy
|       stats\_cfd3d\_mu.npy
|       stats\_cfd3d\_var.npy
|       stats\_dr1d\_mu.npy
|       stats\_dr1d\_var.npy
|       stats\_dr\_mu.npy
|       stats\_dr\_var.npy
|       stats\_fns\_kf\_2d\_mu.npy
|       stats\_fns\_kf\_2d\_var.npy
|       stats\_gsdr2d\_mu.npy
|       stats\_gsdr2d\_var.npy
|       stats\_mhd\_mu.npy
|       stats\_mhd\_var.npy
|       stats\_ns\_gauss\_2d\_mu.npy
|       stats\_ns\_gauss\_2d\_var.npy
|       stats\_ns\_sines\_2d\_mu.npy
|       stats\_ns\_sines\_2d\_var.npy
|       stats\_sw\_mu.npy
|       stats\_sw\_var.npy
|       stats\_tgc3d\_mu.npy
|       stats\_tgc3d\_var.npy
|  
+---datasets
|       **init**.py
|  
+---models
|   ---FM
+---scripts
|       data\_normalization\_revin.py
|       data\_visualization.py
|       finetune\_MORPH.py
|       infer\_MORPH.py
|       pretrain\_MORPH.py
|
---src
|   **init**.py
|  
+---utils
|   |   attention.py
|   |   axial\_attention.py
|   |   axial\_attention\_3dspacetime\_2\_lora.py
|   |   batched\_stream.py
|   |   convert\_nc\_to\_h5.py
|   |   convolutional\_operator.py
|   |   crossattention\_fields.py
|   |   data\_plotter.py
|   |   data\_preparation\_fast.py
|   |   device\_manager.py
|   |   embedding\_conv\_patch\_xatt\_project.py
|   |   explore\_hdf5.py
|   |   explore\_nc.py
|   |   extend\_pos\_embd.py
|   |   feedforward.py
|   |   finetune\_ar1k.py
|   |   lora\_linear.py
|   |   lora\_mha.py
|   |   main\_process\_ddp.py
|   |   metrics\_3d.py
|   |   multi\_source\_iterable\_datasets.py
|   |   normalization.py
|   |   patchify\_3d.py
|   |   positional\_encoding\_spatiotemporal\_bilinear.py
|   |   positional\_encoding\_spatiotemporal\_li\_slice.py
|   |   restrict\_omp.py
|   |   sdpa.py
|   |   select\_fine\_tuning\_parameters.py
|   |   simple\_decoder.py
|   |   stream\_iterabledatasets.py
|   |   test\_rollouts.py
|   |   trainers.py
|   |   transformer\_encoder\_axialattention\_3dspacetime\_lora.py
|   |   visualize\_predictions\_3d\_full.py
|   |   visualize\_rollouts\_3d\_full.py
|   |   vit\_conv\_xatt\_axialatt2.py
|   |  
|   +---dataloaders
|   |       dataloaderchaos.py
|   |       dataloader\_be1d.py
|   |       dataloader\_ce\_2d.py
|   |       dataloader\_cfd1d.py
|   |       dataloader\_cfd2d.py
|   |       dataloader\_cfd2dic.py
|   |       dataloader\_cfd2dic\_fv.py
|   |       dataloader\_cfd3d.py
|   |       dataloader\_cfd3d\_turb.py
|   |       dataloader\_dr.py
|   |       dataloader\_dr1d.py
|   |       dataloader\_fns\_kf\_2d.py
|   |       dataloader\_gsdr2d.py
|   |       dataloader\_mhd.py
|   |       dataloader\_ns\_2d.py
|   |       dataloader\_sw2d.py
|   |       dataloader\_tgc3d.py
|   |  
|   +---datastreamers
|   |   |   datastreamerschaos\_1.py
|   |   |   datastreaming\_be1d\_1.py
|   |   |   datastreaming\_cfd1d\_1.py
|   |   |   datastreaming\_cfd2dic\_1.py
|   |   |   datastreaming\_cfd2dic\_fv.py
|   |   |   datastreaming\_cfd2d\_1.py
|   |   |   datastreaming\_cfd3d\_1.py
|   |   |   datastreaming\_cfd3d\_turb\_1.py
|   |   |   datastreaming\_dr1d\_1.py
|   |   |   datastreaming\_dr\_1.py
|   |   |   datastreaming\_fnskf\_1.py
|   |   |   datastreaming\_gsdr2d\_1.py
|   |   |   datastreaming\_mhd\_1.py
|   |   |   datastreaming\_sw\_1.py
|   |   |   datastreaming\_tgc3d\_1.py
|   |   |  


