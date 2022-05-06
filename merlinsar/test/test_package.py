from merlinsar.train.merlin import create_model, fit_model
from merlinsar.test.spotlight import despeckle_from_crop, despeckle, despeckle_from_coordinates
from merlinsar.train.model import Model
import torch
import numpy as np

# **********************************Train model from scratch*********************************

# nb_epoch=1

# lr = 0.001 * np.ones([nb_epoch])
# lr[6:20] = lr[0]/10
# lr[20:] = lr[0]/100
# # seed=1
# training_set_directory="C:/Users/ykemiche/OneDrive - Capgemini/Desktop/fork/tes/data/training"
# validation_set_directory="C:/Users/ykemiche/OneDrive - Capgemini/Desktop/fork/tes/data/test"
# save_directory="C:/Users/ykemiche/OneDrive - Capgemini/Desktop/fork/tes/save"
# sample_directory="C:/Users/ykemiche/OneDrive - Capgemini/Desktop/fork/tes/sample"
# from_pretrained=False

# model=create_model(batch_size=12,val_batch_size=1,device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),from_pretrained=from_pretrained)
# fit_model(model,lr,nb_epoch,training_set_directory,validation_set_directory,sample_directory,save_directory,seed=2)


# **********************************Fine Tuning*********************************

# nb_epoch=1

# lr = 0.001 * np.ones([nb_epoch])
# lr[6:20] = lr[0]/10
# lr[20:] = lr[0]/100
# # C:\Users\ykemiche\OneDrive - Capgemini\Desktop\fork\tes\data\trainsing
# training_set_directory="C:/Users/ykemiche/OneDrive - Capgemini/Desktop/fork/tes/data/training"
# validation_set_directory="C:/Users/ykemiche/OneDrive - Capgemini/Desktop/fork/tes/data/test"
# save_directory="C:/Users/ykemiche/OneDrive - Capgemini/Desktop/fork/tes/save"
# sample_directory="C:/Users/ykemiche/OneDrive - Capgemini/Desktop/fork/tes/sample"
# from_pretrained=True

# model=create_model(Model,batch_size=12,val_batch_size=1,device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),from_pretrained=from_pretrained)
# fit_model(model,lr,nb_epoch,training_set_directory,validation_set_directory,sample_directory,save_directory,seed=2)


# **********************************Despeckle from crop*********************************


image_path = "C:\\Users\\pblancha\\PycharmProjects\\test_merlin_package\\venv\\IMAGE_HH_SRA_spot_068.cos"
destination_directory = "C:\\Users\\pblancha\\PycharmProjects\\merlinsar\\merlinsar\\test\\results"
model_weights_path = "C:\\Users\\pblancha\\PycharmProjects\\merlinsar\\merlinsar\\test\\saved_model\\model.pth"

# despeckle_from_crop(image_path, destination_directory, stride_size=64,
#                    model_weights_path=model_weights_path, patch_size=256,
#                    height=256,
#                    width=256, fixed=False)

# **********************************Despeckle from coordinates*********************************
coordinates_dict = {}

coordinates_dict["x_start"] = 2600
coordinates_dict["x_end"] = 3000
coordinates_dict["y_start"] = 300
coordinates_dict["y_end"] = 700
#
#
# image_path="C:/Users/ykemiche/OneDrive - Capgemini/Desktop/fork/tes/IMAGE_HH_SRA_spot_068.cos"
# destination_directory="C:/Users/ykemiche/OneDrive - Capgemini/Desktop/fork/tes/results"
# model_weights_path="C:/Users/ykemiche/OneDrive - Capgemini/Desktop/fork/tes/model.pth"
#
despeckle_from_coordinates(image_path, coordinates_dict, destination_directory, model_weights_path=model_weights_path)
#
#

# **********************************Despeckle the entire image*********************************

# image_path="C:/Users/ykemiche/OneDrive - Capgemini/Desktop/fork/tes/IMAGE_HH_SRA_spot_068.cos"
# destination_directory="C:/Users/ykemiche/OneDrive - Capgemini/Desktop/fork/tes/results"
# model_weights_path="C:/Users/ykemiche/OneDrive - Capgemini/Desktop/fork/tes/model.pth"

# despeckle(image_path,destination_directory,model_weights_path=model_weights_path)
