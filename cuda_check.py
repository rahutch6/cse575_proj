import torch
print(torch.__version__)            # should be something like 2.5.1+cu118
print(torch.version.cuda)           # e.g. '11.8'
print(torch.cuda.is_available())    # should now be True
print(torch.device('cuda'))         # cuda:0