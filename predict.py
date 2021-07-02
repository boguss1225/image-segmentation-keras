from keras_segmentation.predict import predict
from keras_segmentation.models.unet import vgg_unet

# load model
# model_weight_path = 'model.h5'
# model = vgg_unet(n_classes=6,  input_height=640, input_width=640)
# model.load_weights(model_weight_path, by_name=True)

# Single Predict 잘작동함
predict( 
    checkpoints_path="checkpoints/mobilenet_segnet", 
    inp="database/IMAS_Salmon/train_images/untitled-10.jpg", 
    out_fname="out_frame/output_SaMobilenet_segnet_Predic.png",
    overlay_img=True
)

# Multi Predict
# predict_multiple( 
#    checkpoints_path="checkpoints/vgg_unet_1", 
#    inp_dir="dataset_path/images_prepped_test/", 
#    out_dir="outputs/" 
# )

# Video Predict
#predict_video(
#    checkpoints_path="checkpoints/vgg_unet_1", 
#    inp="database/Menzies_Brain/val_images/0408.avi", 
#    out_fname="output.avi"
#)