from scOT.model import ScOT
import inspect
from torchinfo import summary
import torch
from transform_to_from_poseidon import to_poseidon_input, undo_poseidon_output

model = ScOT.from_pretrained("camlab-ethz/Poseidon-T")
print("Model:", model)
print("Model params:", sum(p.numel() for p in model.parameters()) / 1e6, "M")
print("Loaded:", type(model))
print("Model type:", getattr(model.config, "model_type", None))
print(inspect.signature(model.forward))

#%% model summary via torchinfo
B, C, H, W = 1, 4, 128, 128
x = torch.zeros(B, C, H, W)
t = torch.zeros(B, 1)  # lead time (dt). If this errors, try torch.zeros(B)

with torch.no_grad():
    print(summary(
        model.eval(),
        input_data={"pixel_values": x, "time": t, "return_dict": False},
        depth=6
    ))

#%% test the to/from poseidon input/output functions
'''
Here we pass a random tensor (PLI shape) through the to_poseidon_input 
and undo_poseidon_output functions to check that the shapes are consistent. 
The actual values won't be meaningful since the model is untrained, 
but this is a sanity check that the data transformations are working as expected.
'''
x1 = torch.randn(1, 1, 1120, 400)          
x4, meta = to_poseidon_input(x1, out_size=128, pad_mode="replicate")
print(x4.shape)  # (B,4,128,128)
t = torch.zeros(x4.shape[0], 1)            # your dt here
with torch.no_grad():
    out = model(pixel_values=x4, time=t, return_dict=False)
y4 = out[0]  
y1_back = undo_poseidon_output(y4, meta, take_channel=0)  # (B,1,1120,400)
print(y1_back.shape)
